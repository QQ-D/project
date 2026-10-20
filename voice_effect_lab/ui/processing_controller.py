"""后台去噪、参数快照与原声/去噪缓存选择。"""
from ..audio.denoise import spectral_subtract
from ..analysis import analyze


class ProcessingController:
    def _init_processing(self):
        self.denoised_clip = None
        self.denoised_analysis = None

    @property
    def current_clip(self):
        return self.denoised_clip if self.stages.checkedId() == 1 else self.clip

    @property
    def stage_name(self):
        return '去噪' if self.stages.checkedId() == 1 else '原声'

    def start_denoise(self):
        if self.occupied:
            return
        if self.clip is None:
            self._error('请先导入或录制原声。')
            return
        if self.noise_clip is None or not self.noise_confirmed:
            self._error('请先导入、采集或选取纯底噪，并确认它适用于当前原声。')
            return
        source, noise, strength = self.clip, self.noise_clip, self.denoise_spin.value()

        def compute():
            result = spectral_subtract(source, noise, strength)
            return result, analyze(result.clip)

        def apply(result):
            if (self.clip is not source or self.noise_clip is not noise
                    or not self.noise_confirmed or self.denoise_spin.value() != strength):
                self._status('输入或参数已变化，本次去噪结果已丢弃，请重新处理。')
                return
            info, analysis = result
            self.denoised_clip, self.denoised_analysis = info.clip, analysis
            details = f'去噪：强度 {strength:g}% · 算法耗时 {info.elapsed_seconds:.3f} s'
            if info.clipped_samples:
                details += f' · {info.clipped_samples} 个越界样本已限幅'
            self.denoise_status_label.setText(details)
            self._set_stage(1)
            self._status('去噪完成；可切换原声/去噪试听，并保存当前阶段 WAV。')

        self._job(compute, apply, '正在后台执行谱减去噪并生成对比图…')

    def _invalidate_denoise(self):
        self.denoised_clip = self.denoised_analysis = None
        self.denoise_status_label.setText('去噪：待处理（输入或参数变化后需重新运行）')
        self._set_stage(0)

    def select_stage(self, stage):
        if self.occupied or stage not in (0, 1):
            return
        if stage == 1 and self.denoised_clip is None:
            self._error('去噪结果尚未生成或已失效，请重新去噪。')
            return
        self._set_stage(stage)
        self._status(f'已选择{self.stage_name}，点击播放从头试听。')

    def _set_stage(self, stage):
        self.stages.button(stage).setChecked(True)
        selected = self.current_clip
        self.player.set_clip(selected)
        if self.clip is not None:
            self.file_label.setText(f'{self.clip.name}  |  频谱/语谱显示至 '
                                    f'{min(8000, self.clip.sample_rate/2):g} Hz  |  当前阶段：{self.stage_name}')
        if self.denoised_clip is None:
            self.plots.clear_result()
        else:
            analysis = self.denoised_analysis if stage == 1 else self.analysis
            self.plots.set_result(selected, analysis, self.stage_name)
        self._position(0)
        self._controls()

    def reset_processing(self):
        if self.occupied:
            return
        self.denoise_spin.setValue(60)
        self._invalidate_denoise()
        self._status('已恢复去噪强度60%并清除去噪结果；原声与底噪保留。')

    def _processing_controls(self):
        idle = not self.occupied
        can_process = idle and self.clip is not None and self.noise_clip is not None and self.noise_confirmed
        self.denoise_button.setEnabled(can_process)
        self.process_button.setEnabled(can_process)
        self.reset_button.setEnabled(idle and self.clip is not None)
        self.denoise_parameters.setEnabled(idle)
        self.stages.button(0).setEnabled(idle and self.clip is not None)
        self.stages.button(1).setEnabled(idle and self.denoised_clip is not None)
        self.save_button.setText(f'保存{self.stage_name} WAV')
