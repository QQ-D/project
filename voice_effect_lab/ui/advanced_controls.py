"""可滚动高级参数、明确的模块顺序及默认状态恢复。"""
from PyQt5.QtCore import Qt,QSignalBlocker,QSize
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QCheckBox,QListWidget,QListWidgetItem,QGridLayout,QFrame,QListView
from ..audio.effects import DEFAULT_CHAIN_ORDER,EFFECT_NAMES

# spin名、Settings字段、上下限、默认、后缀、小数位
PARAMETERS=[('pitch','pitch_semitones',-12,12,0,' 半音',1),('tempo','tempo',.5,2,1,'×',2),
 ('reverb_wet','reverb_wet',0,1,.25,'',2),('reverb_decay','reverb_decay',.2,3,1.2,' s',2),
 ('modulation_frequency','modulation_frequency',20,120,40,' Hz',0),('modulation_mix','modulation_mix',0,1,.6,'',2),
 ('compression_threshold','compression_threshold',-40,-6,-18,' dB',1),('compression_ratio','compression_ratio',1,8,3,' :1',1),
 ('compression_makeup','compression_makeup',0,12,3,' dB',1)]
LABELS={'pitch':'独立音高','tempo':'独立语速','reverb_wet':'混响干湿比','reverb_decay':'混响衰减',
 'modulation_frequency':'调制频率','modulation_mix':'调制混合','compression_threshold':'压缩阈值',
 'compression_ratio':'压缩比','compression_makeup':'压缩补偿'}
GROUPS=[('independent','独立变调 / 变速',['pitch','tempo']),
        ('reverb','房间混响',['reverb_wet','reverb_decay']),
        ('modulation','机械电音 / 环形调制',['modulation_frequency','modulation_mix']),
        ('compression','动态压缩',['compression_threshold','compression_ratio','compression_makeup'])]


class AdvancedControls:
    def build_advanced_panel(self,parameter,button):
        self.advanced_panel=QWidget()
        layout=QVBoxLayout(self.advanced_panel);layout.setContentsMargins(0,0,0,0)
        grid=QGridLayout()
        grid.setSpacing(12)
        for index,(name,title,keys) in enumerate(GROUPS):
            card=QFrame()
            card.setProperty('card',True)
            box=QVBoxLayout(card)
            box.setContentsMargins(14,12,14,12)
            check=QCheckBox(title);setattr(self,name+'_check',check)
            check.setProperty('switch',True)
            check.toggled.connect(self._advanced_toggled)
            box.addWidget(check)
            panel=QWidget();setattr(self,name+'_parameters',panel)
            rows=QVBoxLayout(panel);rows.setContentsMargins(0,6,0,0)
            rows.setSpacing(7)
            for spec in PARAMETERS:
                key,field,low,high,default,suffix,decimals=spec
                if key not in keys:continue
                slider,spin=parameter(rows,LABELS[key],low,high,default,suffix,decimals)
                setattr(self,key+'_slider',slider);setattr(self,key+'_spin',spin)
                spin.valueChanged.connect(self._effect_parameter_changed)
            box.addWidget(panel)
            grid.addWidget(card,index//2,index%2)
        grid.setColumnStretch(0,1)
        grid.setColumnStretch(1,1)
        layout.addLayout(grid)
        note=QLabel('串联顺序：基础倍率先执行，以下仅运行已启用模块。独立变速保持音高，独立变调保持时长。')
        note.setWordWrap(True);layout.addWidget(note)
        self.chain_list=QListWidget();self.chain_list.setFixedHeight(88)
        self.chain_list.setFlow(QListView.LeftToRight)
        self.chain_list.setWrapping(True)
        self.chain_list.setResizeMode(QListView.Adjust)
        self.chain_list.setGridSize(QSize(180,34))
        self.chain_list.setSpacing(3)
        self._restore_chain();layout.addWidget(self.chain_list)
        row=QHBoxLayout();self.chain_up_button=button('↑ 上移');self.chain_down_button=button('↓ 下移')
        self.chain_up_button.clicked.connect(lambda:self.move_effect_step(-1))
        self.chain_down_button.clicked.connect(lambda:self.move_effect_step(1))
        row.addWidget(self.chain_up_button);row.addWidget(self.chain_down_button);layout.addLayout(row)
        self.chain_list.currentRowChanged.connect(self._chain_selection_changed)
        self.advanced_apply_button=button('应用高级音效并返回工作台',True)
        self.advanced_apply_button.clicked.connect(self.start_advanced_effects)
        layout.addWidget(self.advanced_apply_button)
        return self.advanced_panel

    def _advanced_toggled(self,enabled):
        if enabled and not self._loading_preset:self.effects_check.setChecked(True)
        self._effect_parameter_changed()

    def _advanced_settings(self):
        values={field:getattr(self,key+'_spin').value() for key,field,*_ in PARAMETERS}
        values.update({name+'_enabled':getattr(self,name+'_check').isChecked() for name,_,_ in GROUPS})
        values['chain_order']=tuple(self.chain_list.item(i).data(Qt.UserRole) for i in range(self.chain_list.count()))
        return values

    def _restore_chain(self):
        self.chain_list.clear()
        for key in DEFAULT_CHAIN_ORDER:
            item=QListWidgetItem(EFFECT_NAMES[key]);item.setData(Qt.UserRole,key);self.chain_list.addItem(item)
        self.chain_list.setCurrentRow(0)

    def _chain_selection_changed(self,*args):
        row=self.chain_list.currentRow()
        self.chain_up_button.setEnabled(not self.occupied and row>0)
        self.chain_down_button.setEnabled(not self.occupied and 0<=row<self.chain_list.count()-1)

    def move_effect_step(self,direction):
        if self.occupied:return
        row=self.chain_list.currentRow();target=row+direction
        if not 0<=target<self.chain_list.count():return
        item=self.chain_list.takeItem(row);self.chain_list.insertItem(target,item);self.chain_list.setCurrentRow(target)
        self._effect_parameter_changed()

    def start_advanced_effects(self):
        self.start_effects()
        if self.busy:
            self.tabs.setCurrentIndex(0);self.control_scroll.ensureWidgetVisible(self.effects_group,10,10)

    def _advanced_controls(self,available,ready):
        self.advanced_panel.setEnabled(available)
        for name,_,_ in GROUPS:
            getattr(self,name+'_parameters').setEnabled(getattr(self,name+'_check').isChecked())
        self.advanced_apply_button.setEnabled(available and ready)
        self._chain_selection_changed()

    def _reset_advanced_settings(self):
        widgets=[getattr(self,name+'_check') for name,_,_ in GROUPS]+[getattr(self,key+'_spin') for key,*_ in PARAMETERS]
        blockers=[QSignalBlocker(w) for w in widgets]
        for name,_,_ in GROUPS:getattr(self,name+'_check').setChecked(False)
        for key,field,low,high,default,suffix,decimals in PARAMETERS:
            getattr(self,key+'_spin').setValue(default)
            getattr(self,key+'_slider').setValue(round(default*10**decimals))
        self._restore_chain()
        del blockers
