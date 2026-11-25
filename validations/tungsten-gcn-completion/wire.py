"""Losslessly expand row-replicated GCN projections after a device-side bit comparison."""
import numpy as np


def wire_words(p):
    fixed = 18 + p['L']*(p['H']+p['O'])
    return p['R']*(p['C']*fixed + p['K']*p['T'])


def expand(reply, p):
    c, r = p['C'], p['R']
    fixed = 17 + p['L']*(p['H']+p['O'])
    panel = p['K']*p['T']
    if reply.dtype != np.dtype('<u4') or reply.shape != (wire_words(p),):
        raise ValueError('unexpected row-verified frame shape or dtype')
    rows = reply.reshape(r, -1)
    root = rows[:,:fixed+1]
    replica = rows[:,fixed+1:fixed+1+panel]
    peers = rows[:,fixed+1+panel:].reshape(r,c-1,fixed+1)
    if np.any(root[:,-1]) or np.any(peers[:,:,-1]):
        raise ValueError('at least one PE reported a bit difference from its row root')
    records = np.empty((r,c,fixed+panel), dtype='<u4')
    records[:,0,:fixed] = root[:,:fixed]
    records[:,1:,:fixed] = peers[:,:,:fixed]
    records[:,:,fixed:] = replica[:,None,:]
    return records.reshape(r*c,fixed+panel)
