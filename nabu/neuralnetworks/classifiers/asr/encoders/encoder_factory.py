'''@file asr_factory
contains the asr decoder factory'''

from . import listener, ff_listener, listenerCNNBLSTM, listenerACNNBLSTM, listenerACNN


def factory(conf):
    '''create an asr classifier

    Args:
        conf: the classifier config as a dictionary

    Returns:
        An encoder object'''

    if conf['encoder'] == 'listener':
        return listener.Listener(conf)
    if conf['encoder'] == 'ff_listener':
        return ff_listener.FfListener(conf)
    if conf['encoder'] == 'listenercnnblstm':
        return listenerCNNBLSTM.ListenerCNNBLSTM(conf)
    if conf['encoder'] == 'listeneracnnblstm':
        return listenerACNNBLSTM.ListenerACNNBLSTM(conf)
    if conf['encoder'] == 'listeneracnn':
        return listenerACNN.ListenerACNN(conf)
    if conf['encoder'] == 'listenerblstmacnn':
        return listenerBLSTMACNN.ListenerBLSTMACNN(conf)
    else:
        raise Exception('undefined asr encoder type: %s' % conf['encoder'])
