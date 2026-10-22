"""后台去噪、参数快照与原声/去噪缓存选择。"""
from ..audio.denoising import denoise_audio, METHOD_NAMES
from ..analysis import analyze


class ProcessingController:
    def _init_processing(self):
        self.denoised_clip = None
        self.denoised_analysis = None

    @property
    def current_clip(self):
        return {0:self.clip, 1:self.denoised_clip, 2:self.enhanced_clip, 3:self.effected_clip}.get(self.stages.checkedId())

    @property
    def stage_name(self):
        return {0:'原声', 1:'去噪', 2:'增强', 3:'音效'}.get(self.stages.checkedId(), '原声')

    def _can_denoise(self):
        return self.auto_noise_check.isChecked() or (self.noise_clip is not None and self.noise_confirmed)

    def _denoise_snapshot(self):
        automatic = self.auto_noise_check.isChecked()
        return (automatic, self.denoise_spin.value(), None if automatic else id(self.noise_clip),
                True if automatic else self.noise_confirmed, self.denoise_method_combo.currentData())

    def _cache_denoise(self, info, analysis, strength, automatic):
        if self._effects_from_denoise:
            self._clear_effects()
        if self._enhanced_from_denoise:
            self._clear_enhancement()
        self.denoised_clip, self.denoised_analysis = info.clip, analysis
        if info.algorithm == 'deepfilter':
            details = (f'去噪：DeepFilterNet3 · dbaf / CPU · 模型混合 {strength:g}%'
                       f' · 算法耗时 {info.elapsed_seconds:.3f} s')
        else:
            mode = (f'自动估计（{info.estimated_frames}帧，可能含人声）' if info.estimated_frames
                    else '自动估计（0%旁路）') if automatic else '参考底噪'
            tracking = ' · 噪声谱跟踪' if automatic and info.algorithm == 'wiener' and strength else ''
            details = f'去噪：{METHOD_NAMES[info.algorithm]} · {mode}{tracking} · 强度 {strength:g}% · 算法耗时 {info.elapsed_seconds:.3f} s'
        if info.clipped_samples:
            details += f' · {info.clipped_samples} 个越界样本已限幅'
        self.denoise_status_label.setText(details)

    def start_denoise(self):
        if self.occupied:
            return
        if self.clip is None:
            self._error('请先导入或录制原声。')
            return
        automatic = self._denoise_snapshot()[0]
        if not automatic and (self.noise_clip is None or not self.noise_confirmed):
            self._error('请先导入、采集或选取纯底噪，并确认它适用于当前原声。')
            return
        source, strength = self.clip, self.denoise_spin.value()
        snapshot = self._denoise_snapshot()
        method = snapshot[4]
        noise = None if automatic else self.noise_clip

        def compute():
            result = denoise_audio(source, noise, strength, method)
            return result, analyze(result.clip)

        def apply(result):
            if self.clip is not source or self._denoise_snapshot() != snapshot:
                self._status('输入或参数已变化，本次去噪结果已丢弃，请重新处理。')
                return
            info, analysis = result
            self._cache_denoise(info, analysis, strength, automatic)
            self._set_stage(1)
            self._status('去噪完成；可切换原声/去噪试听，并保存当前阶段 WAV。')

        self._job(compute, apply, f'正在后台执行{METHOD_NAMES[method]}并生成对比图…')

    def _noise_mode_changed(self):
        self._invalidate_denoise()
        self._noise_status()

    def _invalidate_denoise(self):
        self.denoised_clip = self.denoised_analysis = None
        if self._effects_from_denoise:
            self._clear_effects()
        if self._enhanced_from_denoise:
            self._clear_enhancement()
        self.denoise_status_label.setText('去噪：待处理（输入或参数变化后需重新运行）')
        self._set_stage(self._valid_stage())

    def select_stage(self, stage):
        if self.occupied or stage not in (0, 1, 2, 3):
            return
        if stage == 3 and self.effected_clip is None:
            self._error('音效结果尚未生成或已失效，请重新应用音效。')
            return
        if stage == 2 and self.enhanced_clip is None:
            self._error('增强结果尚未生成或已失效，请重新增强。')
            return
        if stage == 1 and self.denoised_clip is None:
            self._error('去噪结果尚未生成或已失效，请重新去噪。')
            return
        self._set_stage(stage)
        self._status(f'已选择{self.stage_name}，点击播放从头试听。')

    def _valid_stage(self):
        stages = {0:self.clip, 1:self.denoised_clip, 2:self.enhanced_clip, 3:self.effected_clip}
        current = self.stages.checkedId()
        if stages.get(current) is not None:
            return current
        return next((stage for stage in (2,1,0) if stages[stage] is not None), 0)

    def _set_stage(self, stage):
        self.stages.button(stage).setChecked(True)
        selected = self.current_clip
        self.player.set_clip(selected)
        if self.clip is not None:
            self.file_label.setText(f'{self.clip.name}  |  频谱/语谱显示至 '
                                    f'{min(8000, self.clip.sample_rate/2):g} Hz  |  当前阶段：{self.stage_name}')
        if self.denoised_clip is None and self.enhanced_clip is None and self.effected_clip is None:
            self.plots.clear_result()
        else:
            analysis = {0:self.analysis, 1:self.denoised_analysis, 2:self.enhanced_analysis, 3:self.effected_analysis}[stage]
            self.plots.set_result(selected, analysis, self.stage_name)
        self._position(0)
        self._controls()

    def reset_processing(self):
        if self.occupied:
            return
        self._reset_effect_settings()
        self._reset_enhancement_settings()
        self.denoise_method_combo.setCurrentIndex(self.denoise_method_combo.findData('wiener'))
        self.denoise_spin.setValue(60)
        self.auto_noise_check.setChecked(True)
        self._invalidate_denoise()
        self._status('已恢复默认参数与增强/音效旁路，清除处理结果；原声与参考底噪保留。')

    def _processing_controls(self):
        idle = not self.occupied
        can_process = idle and self.clip is not None and self._can_denoise()
        self.denoise_button.setEnabled(can_process)
        self.process_button.setEnabled(can_process)
        self.reset_button.setEnabled(idle and self.clip is not None)
        self.denoise_parameters.setEnabled(idle)
        self.noise_reference_panel.setVisible(True)
        self.denoise_engine_hint.setVisible(True)
        self.auto_noise_check.setEnabled(idle)
        self.denoise_method_combo.setEnabled(idle)
        self.stages.button(0).setEnabled(idle and self.clip is not None)
        self.stages.button(1).setEnabled(idle and self.denoised_clip is not None)
        self.save_button.setText(f'保存{self.stage_name} WAV')
        self._enhancement_controls()
        self._effects_controls()
