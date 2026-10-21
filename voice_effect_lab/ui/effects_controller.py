"""音效参数、预设与按依赖原子采用的四阶段后台流程。"""
from PyQt5.QtCore import QSignalBlocker

from ..audio.effects import EffectsSettings, apply_effects
from ..audio.enhance import enhance_audio
from ..audio.denoising import denoise_audio
from ..analysis import analyze


class EffectsController:
    def _init_effects(self):
        self.effected_clip = self.effected_analysis = None
        self._effects_from_denoise = self._effects_from_enhancement = False
        self.preset_name = '原声'
        self._loading_preset = False

    def _effect_settings(self):
        return EffectsSettings(self.effects_check.isChecked(), self.rate_spin.value(),
                               self.telephone_check.isChecked(), self.echo_delay_spin.value(),
                               self.echo_strength_spin.value(),
                               int(self.multiecho_count_spin.value()) if self.multiecho_check.isChecked() else 1)

    def _clear_effects(self):
        self.effected_clip = self.effected_analysis = None
        self._effects_from_denoise = self._effects_from_enhancement = False
        self.effects_status_label.setText('音效：待处理')

    def _invalidate_effects(self):
        self._clear_effects()
        self._set_stage(self._valid_stage())

    def _show_preset(self, name):
        self.preset_name = name
        for label, button in self.preset_buttons.items():
            button.setChecked(label == name)
        self.preset_label.setText(f'当前预设：{name} · 加载参数后点击应用音效')

    def _effect_parameter_changed(self):
        if not self._loading_preset:
            self._show_preset('自定义')
        self._invalidate_effects()

    def _enhancement_parameter_changed(self):
        if not self._loading_preset:
            self._show_preset('自定义')
        # 切换增强开关改变音效输入路径；其他设置只影响实际依赖增强的音效。
        if self._effects_from_enhancement or self.enhance_check.isChecked():
            self._clear_effects()
        self._invalidate_enhancement()

    def _upstream_policy_changed(self):
        self._clear_effects()
        self._invalidate_enhancement()

    def select_preset(self, name):
        if self.occupied or name not in self.preset_buttons:
            return
        self._loading_preset = True
        try:
            self.effects_check.setChecked(name != '原声')
            if name != '自定义':
                self.multiecho_check.setChecked(False)
                self.multiecho_count_spin.setValue(3)
                self.rate_spin.setValue(1)
                self.telephone_check.setChecked(name == '电话音')
                self.echo_delay_spin.setValue(.3)
                self.echo_strength_spin.setValue(.45 if name == '山谷回声' else 0)
            voices = {'磁性': (1,4,1,-3), '娃娃音': (1.35,-3,2,3), '大叔音': (.8,4,0,-2)}
            if name in voices:
                speed, low, mid, high = voices[name]
                self.enhance_check.setChecked(True)
                self.clarity_spin.setValue(0)
                self.rate_spin.setValue(speed)
                for spin, value in [(self.eq_low_spin,low),(self.eq_mid_spin,mid),(self.eq_high_spin,high)]:
                    spin.setValue(value)
            self._show_preset(name)
            self._invalidate_effects()
        finally:
            self._loading_preset = False

    def _multiecho_changed(self, enabled):
        if enabled and not self._loading_preset:
            self.effects_check.setChecked(True)
        self._effect_parameter_changed()

    def start_multiecho(self):
        if self.occupied:
            return
        if self.clip is None:
            self._error('请先在基础工作台导入或录制原声。')
            return
        self.multiecho_check.setChecked(True)
        if self.echo_strength_spin.value() == 0:
            self.echo_strength_spin.setValue(.45)
        self.start_effects()
        if self.busy:
            self.tabs.setCurrentIndex(0)
            self.control_scroll.ensureWidgetVisible(self.effects_group,10,10)

    def start_effects(self):
        if self.occupied:
            return
        if self.clip is None:
            self._error('请先导入或录制原声。')
            return
        if not self.effects_check.isChecked():
            self.effects_check.setChecked(True)
        source, settings = self.clip, self._effect_settings()
        enhancement = self._enhancement_settings()
        use_noise = self.use_denoise_check.isChecked()
        if use_noise and self.denoised_clip is None and not self._can_denoise():
            self._error('音效需要去噪输入，请启用自动估计或确认底噪；也可取消“使用去噪结果”。')
            return
        noise_snapshot = self._denoise_snapshot()
        automatic, strength = noise_snapshot[:2]
        method = noise_snapshot[4]
        noise = None if automatic else self.noise_clip
        cached_noise, cached_enhancement = self.denoised_clip, self.enhanced_clip

        def compute():
            new_noise = noise_analysis = new_enhancement = enhancement_analysis = None
            upstream = source
            if use_noise:
                upstream = cached_noise
                if upstream is None:
                    new_noise = denoise_audio(source, noise, strength, method)
                    upstream = new_noise.clip
                    noise_analysis = analyze(upstream)
            if enhancement.enabled:
                upstream = cached_enhancement
                if upstream is None:
                    new_enhancement = enhance_audio(new_noise.clip if new_noise is not None else
                                                   (cached_noise if use_noise else source), enhancement)
                    upstream = new_enhancement.clip
                    enhancement_analysis = analyze(upstream)
            effect = apply_effects(upstream, settings)
            return new_noise, noise_analysis, new_enhancement, enhancement_analysis, effect, analyze(effect.clip)

        def apply(result):
            if (self.clip is not source or self._effect_settings() != settings
                    or self.use_denoise_check.isChecked() != use_noise
                    or self._enhancement_settings() != enhancement
                    or (use_noise and (self._denoise_snapshot() != noise_snapshot
                                       or self.denoised_clip is not cached_noise))
                    or (enhancement.enabled and self.enhanced_clip is not cached_enhancement)):
                self._status('输入或音效参数已变化，本次结果已丢弃，请重新处理。')
                return
            new_noise, noise_analysis, new_enhancement, enhancement_analysis, info, analysis = result
            if new_noise is not None:
                self._cache_denoise(new_noise, noise_analysis, strength, automatic)
            if new_enhancement is not None:
                self._cache_enhancement(new_enhancement, enhancement_analysis, enhancement, use_noise)
            self.effected_clip, self.effected_analysis = info.clip, analysis
            self._effects_from_denoise, self._effects_from_enhancement = use_noise, enhancement.enabled
            source_name = '增强' if enhancement.enabled else ('去噪' if use_noise else '原声')
            details = (f'音效：输入{source_name} · {settings.rate:.2f}× · '
                       f'{info.clip.duration:.2f} s · 算法 {info.elapsed_seconds:.3f} s')
            if settings.echo_strength:
                details += f' · 回声{settings.echo_count}次/间隔{settings.echo_delay:.2f}s'
            if info.clipped_samples:
                details += f' · {info.clipped_samples}个样本限幅，可降低上游增益或回声强度'
            self.effects_status_label.setText(details)
            self._set_stage(3)
            self._status('音效完成，可切换四个阶段试听并保存当前阶段。')

        self._job(compute, apply, '正在后台生成音效及所需的上游结果…')

    def _effects_controls(self):
        idle = not self.occupied
        available = idle and self.clip is not None
        ready = not self.use_denoise_check.isChecked() or self.denoised_clip is not None or self._can_denoise()
        self.multiecho_panel.setEnabled(available)
        self.multiecho_count_controls.setEnabled(self.multiecho_check.isChecked())
        self.multiecho_button.setEnabled(available and ready)
        count = int(self.multiecho_count_spin.value()) if self.multiecho_check.isChecked() else 1
        self.echo_mode_label.setText(f'回声配置：{count}次 · 最后一次延迟 {count*self.echo_delay_spin.value():.2f} s'
                                     if self.echo_strength_spin.value() else '回声配置：强度为0，回声关闭')
        self.effects_check.setEnabled(idle)
        self.effects_parameters.setEnabled(available and self.effects_check.isChecked())
        self.presets_group.setEnabled(available)
        self.effects_button.setEnabled(available and ready)
        self.stages.button(3).setEnabled(idle and self.effected_clip is not None)
        if self.effects_check.isChecked():
            self.process_button.setEnabled(available and ready)

    def _reset_effect_settings(self):
        widgets = [self.effects_check,self.telephone_check,self.rate_spin,
                   self.echo_delay_spin,self.echo_strength_spin,self.multiecho_check,self.multiecho_count_spin]
        blockers = [QSignalBlocker(widget) for widget in widgets]
        self.effects_check.setChecked(False)
        self.telephone_check.setChecked(False)
        self.multiecho_check.setChecked(False)
        self.multiecho_count_spin.setValue(3)
        self.multiecho_count_slider.setValue(3)
        for spin, slider, value in [(self.rate_spin,self.rate_slider,1),
                                     (self.echo_delay_spin,self.echo_delay_slider,.3),
                                     (self.echo_strength_spin,self.echo_strength_slider,0)]:
            spin.setValue(value)
            slider.setValue(round(value*100))
        del blockers
        self.multiecho_delay_spin.setValue(self.echo_delay_spin.value())
        self.multiecho_strength_spin.setValue(self.echo_strength_spin.value())
        self._clear_effects()
        self._show_preset('原声')
