"""主窗口的采集与底噪状态，独立于界面布局。"""
from ..audio.data import AudioError
from ..audio.noise import extract_noise
from ..audio.recorder import AudioRecorder, RecordingError
from ..analysis import analyze
from .noise_dialog import NoiseSelectionDialog


class CaptureController:
    def _init_capture(self, recorder):
        self.recorder = recorder or AudioRecorder(self)
        if self.recorder.parent() is None:
            self.recorder.setParent(self)
        self.noise_clip = None
        self.noise_confirmed = False
        self._noise_kind = None
        self._recording_target = None
        self.recorder.state_changed.connect(self._recording_state)
        self.recorder.duration_changed.connect(self._capture_progress)
        self.recorder.completed.connect(self._capture_completed)
        self.recorder.failed.connect(self._capture_error)

    @property
    def occupied(self):
        return self.busy or self.recorder.state != 'idle'

    def start_recording(self, kind='voice'):
        if self.occupied:
            return
        if kind not in ('voice', 'noise'):
            raise ValueError('未知采集类型')
        self.last_error = ''
        self.player.stop()
        self._recording_target = kind
        limit = 1.5 if kind == 'noise' else 300
        name = '环境底噪.wav' if kind == 'noise' else '麦克风录音.wav'
        try:
            self.recorder.start(sample_rate=int(self.sample_rate_combo.currentData()),
                                limit_seconds=limit, name=name)
        except RecordingError as exc:
            self._capture_error(str(exc))
            return
        if self.recorder.state == 'recording':
            self._status('正在采集底噪，请保持安静；约 1.5 秒后自动结束。' if kind == 'noise'
                         else '正在录音，结束后显示分析图；可点击结束或取消采集。')

    def stop_recording(self):
        self.recorder.stop()

    def cancel_recording(self):
        if not self.recorder.cancel():
            return
        self._recording_target = None
        self.capture_label.setText('本次采集已取消，原有数据保持不变。')
        self._status('已取消采集。')
        self._controls()

    def _recording_state(self, state):
        if hasattr(self, 'capture_label'):
            self._controls()

    def _capture_progress(self, milliseconds):
        kind = '环境底噪' if self._recording_target == 'noise' else '麦克风录音'
        self.capture_label.setText(f'{kind}采集中：{milliseconds/1000:.2f} s')

    def _capture_completed(self, clip):
        kind, self._recording_target = self._recording_target, None
        self.capture_label.setText(f'采集完成：{clip.duration:.2f} s · {clip.sample_rate} Hz')
        if kind == 'noise':
            if clip.duration < .1:
                self._capture_error('底噪至少需要 0.10 秒，请重新采集。')
                return
            self._store_noise(clip, 'microphone')
            self._status('环境底噪已独立保存；可继续录制语音，或对当前原声执行去噪。')
        elif kind == 'voice':
            self._job(lambda: (clip, analyze(clip)),
                      lambda result: self._loaded(result, source_kind='recording'),
                      '录音已结束，正在后台生成原声分析图…')
        self._controls()

    def _capture_error(self, message):
        self._recording_target = None
        self.capture_label.setText('本次采集失败，原有数据保持不变。')
        self._error(message)
        self._controls()

    def _choose_input(self):
        try:
            self.recorder.set_input_device(self.input_combo.currentData())
        except RecordingError as exc:
            self._error(str(exc))
            return
        self.input_combo.setToolTip(self.input_combo.currentText())
        if self.noise_clip is not None:
            self.noise_confirmed = False
            self._invalidate_denoise()
            self._noise_status()
        self._status('输入设备已切换，已有底噪请确认是否仍适用。')
        self._controls()

    def _source_replaced(self, source_kind):
        self._invalidate_denoise()
        if self._noise_kind == 'selection':
            self.noise_clip = None
            self.noise_confirmed = False
            self._noise_kind = None
        elif self.noise_clip is not None and source_kind == 'import':
            self.noise_confirmed = False
        self._noise_status()

    def _store_noise(self, clip, kind):
        self.noise_clip = clip
        self.noise_confirmed = True
        self._noise_kind = kind
        self._invalidate_denoise()
        self._noise_status()
        self._controls()

    def _noise_status(self):
        if self.noise_clip is None:
            self.noise_status_label.setText('底噪：尚未采集或选取')
        else:
            origin = '麦克风' if self._noise_kind == 'microphone' else '原声选段'
            state = '已确认' if self.noise_confirmed else '待确认环境是否匹配'
            self.noise_status_label.setText(f'底噪：{self.noise_clip.duration:.2f} s · '
                                          f'{self.noise_clip.sample_rate} Hz · {origin} · {state}')

    def confirm_noise(self):
        if self.occupied or self.noise_clip is None:
            return
        self.noise_confirmed = True
        self._noise_status()
        self._controls()
        self._status('已确认沿用底噪；请确保它与当前语音来自相同噪声环境。')

    def clear_noise(self):
        if self.occupied:
            return
        self.noise_clip = None
        self.noise_confirmed = False
        self._noise_kind = None
        self._invalidate_denoise()
        self._noise_status()
        self._controls()
        self._status('底噪已清除，原声保持不变。')

    def set_noise_segment(self, start, end, confirmed=False):
        if self.occupied or self.clip is None:
            return False
        if not confirmed:
            self._error('请先确认选段只有环境底噪，没有说话声。')
            return False
        try:
            clip = extract_noise(self.clip, start, end)
        except AudioError as exc:
            self._error(str(exc))
            return False
        self._store_noise(clip, 'selection')
        self._status('已采用纯噪声片段，原声未被裁剪。')
        return True

    def _choose_noise(self):
        if self.occupied or self.clip is None:
            return
        self.player.stop()
        dialog = NoiseSelectionDialog(self.clip, self, self.output_combo.currentData())
        try:
            if dialog.exec_() == dialog.Accepted:
                self._store_noise(dialog.selected_clip, 'selection')
                self._status('已采用纯噪声片段，原声未被裁剪。')
        finally:
            # Qt 父对象会保留隐藏子窗口，显式销毁以释放旧原声和图表引用。
            dialog.deleteLater()

    def _capture_controls(self):
        recording = self.recorder.state == 'recording'
        self.record_button.setEnabled(not self.occupied)
        self.end_record_button.setEnabled(recording)
        self.cancel_capture_button.setEnabled(recording)
        self.noise_button.setEnabled(not self.occupied)
        self.select_noise_button.setEnabled(not self.occupied and self.clip is not None)
        self.input_combo.setEnabled(not self.occupied)
        self.sample_rate_combo.setEnabled(not self.occupied)
        self.confirm_noise_button.setEnabled(not self.occupied and self.noise_clip is not None and not self.noise_confirmed)
        self.clear_noise_button.setEnabled(not self.occupied and self.noise_clip is not None)
        if recording:
            self.stop_button.setEnabled(True)
        self.stop_button.setText('■ 结束录音' if recording else '■ 停止')

    def _stop_action(self):
        if self.recorder.state == 'recording':
            self.stop_recording()
        else:
            self.player.stop()
