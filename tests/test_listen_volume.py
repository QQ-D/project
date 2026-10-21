"""试听音量只更新设备，不改样本或重启播放。"""
import numpy as np
import pytest
from test_player import make_player
from voice_effect_lab.audio.data import AudioClip
from voice_effect_lab.audio.player import PlaybackError


def test_live_volume_and_mute_keep_buffer_position_and_clip(app):
    player,output=make_player(app);source=AudioClip(np.full(8000,.25),8000,'音量.wav')
    try:
        player.set_volume(35);player.set_clip(source);player.play()
        assert output.volume==.35
        output.consume(800);position=player.position_ms;buffer=player._buffer
        player.set_volume(70)
        assert output.volume==.7 and player.state=='playing'
        assert player.position_ms==position and player._buffer is buffer
        player.pause();player.set_muted(True)
        assert output.volume==0 and player.volume==70 and player.muted
        player.set_muted(False)
        assert output.volume==.7 and player.state=='paused'
        player.stop();player.play()
        assert output.volume==.7 and player._clip is source
        np.testing.assert_array_equal(source.samples,np.full(8000,.25))
    finally:player.close()


@pytest.mark.parametrize('value',[-1,101,np.nan,np.inf,True,'50'])
def test_invalid_listen_volume_is_rejected(app,value):
    player,_=make_player(app)
    try:
        with pytest.raises(PlaybackError):player.set_volume(value)
    finally:player.close()
