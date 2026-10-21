"""Independent TTS cache and playback, using the host's global background job.

Integration: call _init_tts() before building controls, insert build_tts_panel(),
and call _tts_controls() from the host's _controls(). Stop tts_player before host
playback, capture/jobs and close it on shutdown. start_tts(), save_tts_to(path),
play_tts() and send_tts_to_workbench() are the public action methods. The host's
_job owns task exclusion/failure reporting; _loaded uses normal import semantics.
"""
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog

from ..analysis import analyze
from ..audio.data import save_wav
from ..audio.player import AudioPlayer, PlaybackError
from ..audio.tts import MAX_CHARACTERS, synthesize_chinese, tts_available
from .tts_panel import make_tts_panel


class TtsController:
    def _init_tts(self):
        self.tts_clip = self.tts_analysis = None
        self._tts_revision = 0
        self.tts_player = AudioPlayer(self)
        self.tts_player.state_changed.connect(self._tts_playback_state)
        self.tts_player.failed.connect(self._error)
        self.tts_player.position_changed.connect(self._tts_position)
        self.tts_player.output_device_changed.connect(self._tts_output_fallback)
        self._tts_available, self._tts_engine_message = tts_available()

    def build_tts_panel(self):
        if not hasattr(self, 'tts_panel'):
            self.tts_panel = make_tts_panel(self)
        return self.tts_panel

    def _tts_snapshot(self):
        return self.tts_text.toPlainText(), self.tts_speed_spin.value(), self._tts_revision

    def _tts_changed(self):
        self._tts_revision += 1
        self.tts_clip = self.tts_analysis = None
        self.tts_player.set_clip(None)
        self.tts_result_label.setText('文字或语速已变化，请重新生成。')
        self.tts_count_label.setText(f'{len(self.tts_text.toPlainText())} / {MAX_CHARACTERS} 字符')
        self._tts_controls()

    def start_tts(self):
        if self.occupied:
            return
        if not self._tts_available:
            self._error(self._tts_engine_message)
            return
        text, speed, revision = snapshot = self._tts_snapshot()
        if not text.strip() or len(text) > MAX_CHARACTERS:
            self._error(f'请输入 1–{MAX_CHARACTERS} 个字符的中文文字。')
            return

        def compute():
            clip = synthesize_chinese(text, speed)
            return clip, analyze(clip)

        def apply(result):
            if self._tts_snapshot() != snapshot:
                self._status('文字或合成语速已变化，本次中文合成结果已丢弃。')
                return
            self.tts_clip, self.tts_analysis = result
            self.tts_player.set_clip(self.tts_clip)
            self.tts_result_label.setText(f'合成完成 · {self.tts_clip.sample_rate} Hz · '
                                          f'{self.tts_clip.duration:.2f} s · 语速 {speed:.2f}×')
            self._status('中文合成完成，可独立试听、保存或点击送入工作台。')

        self._job(compute, apply, '正在离线合成中文并计算分析数据…')

    def play_tts(self):
        if self.occupied or self.tts_clip is None:
            return
        self.player.stop()
        try:
            self.tts_player.set_output_device(self.output_combo.currentData())
            self.tts_player.play()
        except PlaybackError as exc:
            self._error(str(exc))

    def save_tts_to(self, path):
        if self.occupied or self.tts_clip is None or not path:
            return
        if Path(path).suffix.lower() != '.wav':
            self._error('请使用 .wav 扩展名保存中文合成音频。')
            return
        clip = self.tts_clip
        self._job(lambda: save_wav(path, clip),
                  lambda _: self._status(f'已保存中文合成 WAV：{path}'),
                  '正在保存中文合成 WAV…')

    def _choose_tts_save(self):
        if self.occupied or self.tts_clip is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, '保存中文合成', '中文合成.wav', 'WAV 音频 (*.wav)')
        if path:
            self.save_tts_to(path)

    def send_tts_to_workbench(self):
        if self.occupied or self.tts_clip is None:
            return
        self.tts_player.stop()
        self.player.stop()
        # Default _loaded source_kind='import' resets the source, processing
        # caches and any prior reference-noise confirmation in the host.
        self._loaded((self.tts_clip, self.tts_analysis))
        self.tabs.setCurrentIndex(0)
        self._controls()
        self._status('中文合成已送入工作台作为原声，可继续去噪、增强和制作音效。')

    def _tts_controls(self):
        if not hasattr(self, 'tts_generate_button'):
            return
        idle = not self.occupied
        ready = idle and self.tts_clip is not None
        text = self.tts_text.toPlainText()
        self.tts_text.setEnabled(idle)
        self.tts_speed_spin.setEnabled(idle and self._tts_available)
        self.tts_speed_slider.setEnabled(idle and self._tts_available)
        self.tts_generate_button.setEnabled(idle and self._tts_available
                                             and bool(text.strip()) and len(text) <= MAX_CHARACTERS)
        self.tts_play_button.setEnabled(ready and self.tts_player.state != 'playing')
        self.tts_stop_button.setEnabled(ready and self.tts_player.state in ('playing', 'paused', 'finished'))
        self.tts_save_button.setEnabled(ready)
        self.tts_send_button.setEnabled(ready)

    def _tts_playback_state(self, state):
        self._tts_controls()
        if state == 'playing':
            self._status('正在试听中文合成。')
        elif state == 'finished':
            self._status('中文合成试听结束，可再次试听或送入工作台。')

    def _tts_position(self, position):
        if hasattr(self, 'tts_progress_label'):
            self.tts_progress_label.setText(f'{position/1000:.2f} / {self.tts_player.duration_ms/1000:.2f} s')

    def _tts_output_fallback(self, name):
        # Host helper updates the shared selector without stopping playback.
        if hasattr(self, '_output_fallback'):
            self._output_fallback(name)
