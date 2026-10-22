"""增强参数与后台链路；仅在成功且快照有效后采用缓存。"""
from PyQt5.QtCore import QSignalBlocker

from ..audio.enhance import EnhancementSettings, enhance_audio
from ..audio.denoising import denoise_audio
from ..analysis import analyze


class EnhancementController:
    def _init_enhancement(self):
        self.enhanced_clip = None
        self.enhanced_analysis = None
        self._enhanced_from_denoise = False

    def _enhancement_settings(self):
        return EnhancementSettings(
            enabled=self.enhance_check.isChecked(),
            normalize_peak=self.normalize_check.isChecked(),
            gain=self.gain_spin.value(), clarity=self.clarity_spin.value(),
            low_db=self.eq_low_spin.value(), mid_db=self.eq_mid_spin.value(),
            high_db=self.eq_high_spin.value())

    def _clear_enhancement(self):
        if self._effects_from_enhancement:
            self._clear_effects()
        self.enhanced_clip = self.enhanced_analysis = None
        self._enhanced_from_denoise = False
        self.enhance_status_label.setText('增强：待处理')

    def _invalidate_enhancement(self):
        self._clear_enhancement()
        self._set_stage(self._valid_stage())

    def start_processing(self):
        if self.effects_check.isChecked():
            self.start_effects()
        elif self.enhance_check.isChecked():
            self.start_enhance()
        else:
            self.start_denoise()

    def _cache_enhancement(self, info, analysis, settings, use_denoise):
        if self._effects_from_enhancement:
            self._clear_effects()
        self.enhanced_clip, self.enhanced_analysis = info.clip, analysis
        self._enhanced_from_denoise = use_denoise
        details = (f'增强：输入{"去噪" if use_denoise else "原声"} · '
                   f'峰值 {info.input_peak:.3f} → {info.output_peak:.3f} · '
                   f'增益 {settings.gain:.2f}× · {info.elapsed_seconds:.3f} s')
        if info.clipped_samples:
            details += f' · {info.clipped_samples}个样本限幅，可降低增益'
        self.enhance_status_label.setText(details)

    def start_enhance(self):
        if self.occupied:
            return
        if self.clip is None:
            self._error('请先导入或录制原声。')
            return
        self.enhance_check.setChecked(True)
        use_denoise = self.use_denoise_check.isChecked()
        cached = self.denoised_clip
        if use_denoise and cached is None and not self._can_denoise():
            self._error('增强需要去噪输入，请启用自动估计或确认底噪；也可取消“使用去噪结果”。')
            return
        source, settings = self.clip, self._enhancement_settings()
        noise_snapshot = self._denoise_snapshot()
        automatic, strength = noise_snapshot[:2]
        method = noise_snapshot[4]
        noise = None if automatic else self.noise_clip

        def compute():
            new_noise = None
            noise_analysis = None
            upstream = source
            if use_denoise:
                upstream = cached
                if upstream is None:
                    new_noise = denoise_audio(source, noise, strength, method)
                    upstream = new_noise.clip
                    noise_analysis = analyze(upstream)
            enhanced = enhance_audio(upstream, settings)
            return new_noise, noise_analysis, enhanced, analyze(enhanced.clip)

        def apply(result):
            if (self.clip is not source or self._enhancement_settings() != settings
                    or self.use_denoise_check.isChecked() != use_denoise
                    or (use_denoise and (self._denoise_snapshot() != noise_snapshot
                                         or self.denoised_clip is not cached))):
                self._status('输入或增强参数已变化，本次结果已丢弃，请重新处理。')
                return
            new_noise, noise_analysis, info, analysis = result
            if new_noise is not None:
                self._cache_denoise(new_noise, noise_analysis, strength, automatic)
            self._cache_enhancement(info, analysis, settings, use_denoise)
            self._set_stage(2)
            self._status('增强完成，可切换原声/去噪/增强试听或保存当前阶段。')

        self._job(compute, apply, '正在后台生成增强结果与分析图…')

    def _enhancement_controls(self):
        self.enhance_status_label.setVisible(self.enhanced_clip is not None)
        idle = not self.occupied
        available = idle and self.clip is not None
        ready = not self.use_denoise_check.isChecked() or self.denoised_clip is not None or self._can_denoise()
        self.enhance_button.setEnabled(available and ready)
        self.enhance_check.setEnabled(idle)
        self.normalize_check.setEnabled(available and self.enhance_check.isChecked())
        self.use_denoise_check.setEnabled(available)
        self.enhance_parameters.setEnabled(available and self.enhance_check.isChecked())
        self.stages.button(2).setEnabled(idle and self.enhanced_clip is not None)
        if self.enhance_check.isChecked():
            self.process_button.setEnabled(available and ready)

    def _reset_enhancement_settings(self):
        widgets = [self.enhance_check, self.normalize_check, self.use_denoise_check,
                   self.gain_spin, self.clarity_spin, self.eq_low_spin, self.eq_mid_spin, self.eq_high_spin]
        blockers = [QSignalBlocker(widget) for widget in widgets]
        self.enhance_check.setChecked(False)
        self.normalize_check.setChecked(True)
        self.use_denoise_check.setChecked(False)
        for spin, value in [(self.gain_spin,1),(self.clarity_spin,0),
                            (self.eq_low_spin,0),(self.eq_mid_spin,0),(self.eq_high_spin,0)]:
            spin.setValue(value)
        for slider, spin in [(self.gain_slider,self.gain_spin),(self.clarity_slider,self.clarity_spin),
                             (self.eq_low_slider,self.eq_low_spin),(self.eq_mid_slider,self.eq_mid_spin),
                             (self.eq_high_slider,self.eq_high_spin)]:
            slider.setValue(round(spin.value()*10**spin.decimals()))
        del blockers
        self._clear_enhancement()
