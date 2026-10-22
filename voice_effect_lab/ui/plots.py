"""从实际选中缓存显示原声与处理结果，使用同一幅度/频率色标。"""
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from . import chart_style  # 所有图表使用同一中文字体配置。


SPECTROGRAM_MAP = colors.LinearSegmentedColormap.from_list(
    'voice_studio', ['#08265d','#1166ad','#08a9d2','#19d2b6','#95e35e','#ffdc40'])


class AnalysisPlots(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 8), dpi=100, layout='constrained', facecolor='white')
        self.axes = self.figure.subplots(3, 2)
        super().__init__(self.figure)
        self.setParent(parent)
        self._colors = ('#1688e8', '#ed9040')
        self._style()
        mappable = ScalarMappable(norm=colors.Normalize(-100, 0), cmap=SPECTROGRAM_MAP)
        self.colorbar = self.figure.colorbar(mappable, ax=list(self.axes[2]), orientation='horizontal', shrink=.78, pad=.09, aspect=55)
        self.colorbar.set_label('幅度（dBFS）· 共用色标', fontsize=8, color='#6a7e93')
        self.colorbar.outline.set_visible(False)
        self.colorbar.ax.tick_params(labelsize=8, colors='#6a7e93', length=0)
        self.figure.get_layout_engine().set(w_pad=.06, h_pad=.055, wspace=.07, hspace=.07)

    def _style(self):
        self._style_column(0, '原声', '请导入 WAV')
        self._style_column(1, '处理结果（待处理）', '载入原声后可直接去噪或增强')

    def _style_column(self, col, name, message=''):
        titles = ['波形', '幅度频谱', '语谱图']
        for row in range(3):
            ax = self.axes[row, col]
            ax.clear()
            ax.set_title(f'{name} · {titles[row]}', fontsize=10, pad=12, loc='left', fontweight='medium', color='#294158')
            ax.tick_params(labelsize=8, colors='#6e8197', length=0, pad=4)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            for spine in ax.spines.values():
                spine.set_color('#e0e8ef')
                spine.set_linewidth(.7)
            if row == 2:
                ax.grid(False)
            else:
                ax.grid(True, color='#edf1f5', linewidth=.6)
            ax.set_axisbelow(True)
            ax.set_xlabel('频率（Hz）' if row == 1 else '时间（s）', fontsize=8)
            ax.set_ylabel(['幅度', '幅度（dBFS）', '频率（Hz）'][row], fontsize=8)
            if row == 0:
                ax.set_ylim(-1, 1)
                ax.set_yticks([-1,-.5,0,.5,1])
            if row == 1:
                ax.set_ylim(-120, 0)
            if message:
                ax.text(.5, .5, message, transform=ax.transAxes, ha='center',
                        va='center', color='#9aa7b8', fontsize=8)

    def set_source(self, clip, result):
        self._style()
        for ax in self.axes[:, 0]:
            for text in list(ax.texts):
                text.remove()
        self._draw_column(0, clip, result, '原声')
        self.draw_idle()

    def clear_result(self):
        self._style_column(1, '处理结果（待处理）', '去噪或增强后对比；输入、模式或参数变化后需重新处理')
        self.draw_idle()

    def set_result(self, clip, result, name='去噪'):
        self._style_column(1, name)
        self._draw_column(1, clip, result, name)
        self.draw_idle()

    def _draw_column(self, col, clip, result, name):
        duration = max(clip.duration, .01)
        ax = self.axes[0, col]
        ax.plot(result.wave_times, result.wave_values, color=self._colors[col], linewidth=.65,
                marker='.' if len(result.wave_values) < 5 else None)
        ax.set_xlim(0, duration)
        freq_limit = min(8000, clip.sample_rate/2)
        ax = self.axes[1, col]
        ax.plot(result.frequencies, result.spectrum_db, color=self._colors[col], linewidth=.7)
        ax.set_xlim(0, freq_limit)
        ax.fill_between(result.frequencies, result.spectrum_db, -120, color=self._colors[col], alpha=.09, linewidth=0)
        ax.set_title(f'{name} · 幅度频谱（前 {result.spectrum_duration:.2f} s）', fontsize=10, pad=12, loc='left', fontweight='medium', color='#294158')
        ax = self.axes[2, col]
        # 统一时间网格的图像显示；长录音的 STFT 帧被均匀抽样以控制内存。
        ax.imshow(result.spectrogram_db, origin='lower', aspect='auto', cmap=SPECTROGRAM_MAP,
                  extent=(0, duration, 0, clip.sample_rate/2), vmin=-100, vmax=0,
                  interpolation='nearest')
        ax.set_xlim(0, duration)
        ax.set_ylim(0, freq_limit)
