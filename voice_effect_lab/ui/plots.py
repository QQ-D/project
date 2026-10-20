"""仅显示真实缓存分析结果；未实现的处理结果保持空白。"""
from matplotlib import colormaps, colors
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from . import chart_style  # 所有图表使用同一中文字体配置。


class AnalysisPlots(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 8), dpi=100, layout='constrained', facecolor='white')
        self.axes = self.figure.subplots(3, 2)
        super().__init__(self.figure)
        self.setParent(parent)
        self._colors = ('#1685f8', '#f47729')
        self._style()
        mappable = ScalarMappable(norm=colors.Normalize(-100, 0), cmap=colormaps['turbo'])
        self.colorbar = self.figure.colorbar(mappable, ax=list(self.axes[2]), orientation='horizontal', shrink=.78, pad=.09, aspect=55)
        self.colorbar.set_label('幅度（dBFS）· 共用色标', fontsize=9)

    def _style(self):
        titles = ['波形', '幅度频谱', '语谱图']
        for row in range(3):
            for col in range(2):
                ax = self.axes[row, col]
                ax.clear()
                name = '原声' if col == 0 else '处理结果（待实现）'
                ax.set_title(f'{name} · {titles[row]}', fontsize=10, pad=10)
                ax.tick_params(labelsize=8, colors='#64748b')
                for spine in ax.spines.values():
                    spine.set_color('#cfdae6')
                ax.grid(True, color='#e7edf5', linewidth=.6)
                ax.set_axisbelow(True)
                ax.set_xlabel('频率（Hz）' if row == 1 else '时间（s）', fontsize=8)
                ax.set_ylabel(['幅度', '幅度（dBFS）', '频率（Hz）'][row], fontsize=8)
                if row == 0:
                    ax.set_ylim(-1, 1)
                if row == 1:
                    ax.set_ylim(-120, 0)
                ax.text(.5, .5, '请导入 WAV' if col == 0 else '去噪 / 增强 / 音效将在后续阶段接入',
                        transform=ax.transAxes, ha='center', va='center', color='#9aa7b8', fontsize=8)

    def set_source(self, clip, result):
        self._style()
        for ax in self.axes[:, 0]:
            for text in list(ax.texts):
                text.remove()
        duration = max(clip.duration, .01)
        ax = self.axes[0, 0]
        ax.plot(result.wave_times, result.wave_values, color=self._colors[0], linewidth=.65,
                marker='.' if len(result.wave_values) < 5 else None)
        ax.set_xlim(0, duration)
        freq_limit = min(8000, clip.sample_rate/2)
        ax = self.axes[1, 0]
        ax.plot(result.frequencies, result.spectrum_db, color=self._colors[0], linewidth=.7)
        ax.set_xlim(0, freq_limit)
        ax.set_title(f'原声 · 幅度频谱（前 {result.spectrum_duration:.2f} s）', fontsize=10, pad=10)
        ax = self.axes[2, 0]
        # 统一时间网格的图像显示；长录音的 STFT 帧被均匀抽样以控制内存。
        ax.imshow(result.spectrogram_db, origin='lower', aspect='auto', cmap='turbo',
                  extent=(0, duration, 0, clip.sample_rate/2), vmin=-100, vmax=0,
                  interpolation='nearest')
        ax.set_xlim(0, duration)
        ax.set_ylim(0, freq_limit)
        self.draw_idle()
