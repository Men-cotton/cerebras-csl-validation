// Two-layer full-neighbor mean GraphSAGE, resident FP32 inference.
import arch_hdr;

xp param C;
xp param R;
xp param K;
xp param L;
xp param FW;
xp param H;
xp param O;
xp param T;
xp param MX;
xp param MG;
xp param U;
xp param COMPACT;
xp param X0 = 0;
xp param Y0 = 0;

xp const OWNER_MAJOR=1;
xp param NTX;
xp param NRX;
socket mp;
socket east;
socket host_in;
socket host_out;
socket rr;                  // receive right, send left
socket rb;                  // left-root row broadcast
socket yp;                  // send to y+1, receive from y-1
socket ym;                  // send to y-1, receive from y+1

u16 x_ptr[n=K+1];
u16 x_col[n=MX];
sp x_val[n=MX];
// Packed by output tile, input feature, neighbor/self bank and tile lane.
sp paired0[n=((H+T-1)/T)*FW*2*T];
sp paired1[n=((O+T-1)/T)*H*2*T];
sp b0[n=H];
sp b1[n=O];
u16 slots[n=L];
u16 counts[n=2];
u32 tx_hdr[n=NTX];
u16 tx_tag[n=NTX];
u16 tx_slot[n=NTX];
u16 edge_ptr[n=L+1];
u16 edge_src[n=MG];
sp edge_weight[n=L];

sp pre[n=L*H];
sp hidden[n=L*H];
sp logits[n=L*O];
// Self and neighbor features share one row reduction per output tile.
sp panel[n=2*K*T];
// Row reduction uses the idle delivery buffer.
u32 envelope[n=NTX*(2+T)];
sp payloads[n=(L+NRX)*T];
sp accum[n=T];
u16 clocks[n=6];
u16 done[n=1];
u16 row_id[1] address(0x7c45);
u16 col_id[1] address(0x7c46);
u16 tsc[3] address(CE_TSC_ADDR);
u16 tsc_control[1] address(CE_PERF_CNTR_CTRL_ADDR);

function clock_read(xp offset) {
    // Low-word-first read latches the remaining 48-bit timestamp words.
    clocks[offset] = tsc[0];
    clocks[offset+1] = tsc[1];
    clocks[offset+2] = tsc[2];
}

function barrier(xp x, xp y) {
    // Two-dimensional reduction/broadcast, outside measured GCN intervals.
    sp token = 0.0;
    if (x < C-1) token = rr[];
    if (x > 0) rr[] <- token;
    if (x == 0) rb[] <- token;
    else token = rb[];
    if (y < R-1) token = ym[];
    if (y > 0) ym[] <- token;
    if (y == 0) yp[] <- token;
    else {
        token = yp[];
        if (y < R-1) yp[] <- token;
    }
}

// All remote consumers of a projected source on one PE share a receive slot.
function deliver(xp x, xp y, xp previous) {
    xp send_count = counts[0];
    xp recv_count = counts[1];
    v in [0,L) {
        if (slots[v] != 65535) {
            xp src = slots[v]*2*T;
            xp dst = v*T;
            i in [0,T) payloads[dst+i] <- panel[src+i];
        }
    }
    j in [0,send_count) {
        xp base = j*(2+T);
        xp src = tx_slot[j]*2*T;
        envelope[base] = tx_hdr[j];
        envelope[base+1] = (u32)tx_tag[j];
        i in [0,T) envelope[base+2+i] <- bitcast(u32,panel[src+i]);
    }
    if (previous > 0) barrier(x,y);
    parallel {
        { if (send_count > 0) {
            i in [0,send_count*(2+T)) mp[] <- envelope[i];
        } }
        { if (recv_count > 0) {
            j in [0,recv_count) {
                u32 hdr;
                u32 tag;
                hdr <- mp[];
                tag <- mp[];
                xp dst = ((xp)tag)*T;
                i in [0,T) payloads[dst+i] <- mp[];
            }
        } }
    }
}

function from_left(xp x) {
    xp block=L*2*T;
    xp start=(x+1)*block;
    xp onward=(C-x-1)*block;
    if (x==0) {
        i in [0,onward) east[] <- panel[start+i];
    } else {
        xp own=x*block;
        i in [0,block) panel[own+i] <- east[]+panel[own+i];
        if (x<C-1) i in [0,onward) east[] <- east[]+panel[start+i];
    }
}
function from_right(xp x) {
    xp block=L*2*T;
    xp prefix=x*block;
    if (x==C-1) {
        i in [0,prefix) rr[] <- panel[i];
    } else {
        if (x>0) i in [0,prefix) rr[] <- rr[]+panel[i];
        i in [0,block) payloads[i] <- rr[];
    }
}
function row_sum(xp x,xp count) {
    parallel {
        { from_left(x); }
        { from_right(x); }
    }
    if (x<C-1) {
        xp start=x*L*2*T;
        i in [0,L*2*T) panel[start+i] <- panel[start+i]+payloads[i];
    }
}

function normalize() {
    v in [0, L) {
        xp base = v*H;
        sp scale = 0.0;
        j in [0, H) {
            sp pos = pre[base+j];
            if (!(pos > 0.0)) pos = 0.0;
            hidden[base+j] = pos;
            if (pos > scale) scale = pos;
        }
        if (scale > 0.0) {
            sp squares = 0.0;
            if (scale >= 1.0e-12 && scale <= 1.0e12) {
                j in [0, H) squares <+- hidden[base+j] * hidden[base+j];
            } else {
                sp inv_scale = 0.0;
                if (scale >= 1.17549435e-38) inv_scale = 1.0 / scale;
                j in [0, H) {
                    sp value = hidden[base+j];
                    if (scale >= 1.17549435e-38) value = value * inv_scale;
                    else value = value / scale;
                    hidden[base+j] = value;
                    squares <+- value * value;
                }
            }
            // Both paths produce a positive normal sum of squares. Use a bit seed
            // and three Newton iterations for FP32 reciprocal square root.
            u32 bits = 0x5f3759df - (bitcast(u32, squares) >> 1);
            sp inv = bitcast(sp, bits);
            j in [0, 3) inv = inv * (1.5 - (0.5*squares*inv)*inv);
            j in [0, H) hidden[base+j] <- hidden[base+j] * inv;
        }
    }
}

// Only PE (0,0) consumes Direct Link input. Distribute the frame's FP32
// scale over the existing first column and then each compute row.
function frame_scale(xp x, xp y) -> sp {
    sp scale = 0.0;
    if (x == 0) {
        if (y == 0) scale <- host_in[];
        else scale <- yp[];
        if (y < R-1) yp[] <- scale;
        rb[] <- scale;
    } else scale <- rb[];
    return scale;
}

function stream_rr(xp x, xp y, xp trial) {
    xp words = 10+L*(2*H+O);
    rr[] <- (u32)trial;
    rr[] <- (u32)x;
    rr[] <- (u32)y;
    i in [0,6) rr[] <- (u32)clocks[i];
    rr[] <- (u32)done[0];
    i in [0,L*H) rr[] <- pre[i];
    i in [0,L*H) rr[] <- hidden[i];
    i in [0,L*O) rr[] <- logits[i];
    c in [x+1,C) { i in [0,words) rr[] <- (u32)rr[]; }
}

function stream_yp(xp x, xp y, xp trial) {
    xp words = 10+L*(2*H+O);
    yp[] <- (u32)trial;
    yp[] <- (u32)x;
    yp[] <- (u32)y;
    i in [0,6) yp[] <- (u32)clocks[i];
    yp[] <- (u32)done[0];
    i in [0,L*H) yp[] <- pre[i];
    i in [0,L*H) yp[] <- hidden[i];
    i in [0,L*O) yp[] <- logits[i];
    c in [x+1,C) { i in [0,words) yp[] <- (u32)rr[]; }
}

function stream_host_out(xp x, xp y, xp trial) {
    xp words = 10+L*(2*H+O);
    host_out[] <- (u32)trial;
    host_out[] <- (u32)x;
    host_out[] <- (u32)y;
    i in [0,6) host_out[] <- (u32)clocks[i];
    host_out[] <- (u32)done[0];
    i in [0,L*H) host_out[] <- pre[i];
    i in [0,L*H) host_out[] <- hidden[i];
    i in [0,L*O) host_out[] <- logits[i];
    c in [x+1,C) { i in [0,words) host_out[] <- (u32)rr[]; }
}

function stream_ym(xp x, xp y, xp trial) {
    xp words = 10+L*(2*H+O);
    ym[] <- (u32)trial;
    ym[] <- (u32)x;
    ym[] <- (u32)y;
    i in [0,6) ym[] <- (u32)clocks[i];
    ym[] <- (u32)done[0];
    i in [0,L*H) ym[] <- pre[i];
    i in [0,L*H) ym[] <- hidden[i];
    i in [0,L*O) ym[] <- logits[i];
    c in [x+1,C) { i in [0,words) ym[] <- (u32)rr[]; }
}
function readout(xp x, xp y, xp trial) {
    xp words = 10+L*(2*H+O);
    if (x > 0) stream_rr(x,y,trial);
    else if (y == 0) stream_yp(x,y,trial);
    else if (y == 1) {
        c in [0,C) { i in [0,words) host_out[] <- (u32)yp[]; }
        stream_host_out(x,y,trial);
        r in [2,R) { c in [0,C) { i in [0,words) host_out[] <- (u32)ym[]; } }
    } else {
        stream_ym(x,y,trial);
        r in [y+1,R) { c in [0,C) { i in [0,words) ym[] <- (u32)ym[]; } }
    }
}

main {
    xp x = col_id[0] - X0;
    xp y = row_id[0] - Y0;
    tsc_control[0] = tsc_control[0] | 4;
    xp trial = 1;
    while (trial <= 3) {
    sp scale = frame_scale(x,y);
    i in [0,MX) x_val[i] <- x_val[i]*scale;
    done[0] = 0;
    i in [0, L*H) pre[i] <- 0.0;
    i in [0, L*O) logits[i] <- 0.0;
    barrier(x,y);
    clock_read(0);
    phase in [0, 2) {
        xp width = H;
        if (phase == 1) width = O;
        xp lane = 0;
        while (lane < width) {
            xp active = T;
            if (lane+T > width) active = width-lane;
            i in [0, 2*K*T) panel[i] <- 0.0;
            if (phase == 0) {
                v in [0, K) {
                    xp dst = v*2*T;
                    xp tile_bank = (lane/T)*FW;
                    xp e = x_ptr[v];
                    while (e < x_ptr[v+1]) {
                        xp src = (tile_bank+x_col[e])*2*T;
                        sp value = x_val[e];
                        i in [0, T+active) panel[dst+i] <+- paired0[src+i] * value;
                        e = e+1;
                    }
                }
            } else {
                v in [0, L) {
                    if (slots[v] != 65535) {
                        xp dst = slots[v]*2*T;
                        xp tile_bank = (lane/T)*H;
                        j in [0, H) {
                            xp src = (tile_bank+j)*2*T;
                            sp value = hidden[v*H+j];
                            if (value != 0.0) { i in [0, T+active) panel[dst+i] <+- paired1[src+i] * value; }
                        }
                    }
                }
            }
            if (phase == 0) row_sum(x, 2*K*T);
            deliver(x,y,phase+lane);
            v in [0,L) {
                if (slots[v] != 65535) {
                    xp self = slots[v]*2*T+T;
                    xp dst = v*width+lane;
                    xp e = edge_ptr[v];
                    xp end = edge_ptr[v+1];
                    if (e < end) {
                        xp first = edge_src[e]*T;
                        sp w = edge_weight[v];
                        e = e+1;
                        if (e < end) {
                            i in [0,active) accum[i] <- payloads[first+i];
                            while (e < end) {
                                xp src = edge_src[e]*T;
                                i in [0,active) accum[i] <+- payloads[src+i];
                                e = e+1;
                            }
                            i in [0,active) panel[self+i] <+- accum[i]*w;
                        } else { i in [0,active) panel[self+i] <+- payloads[first+i]; }
                    }
                    if (phase == 0) {
                        i in [0,active) pre[dst+i] <- panel[self+i]+b0[lane+i];
                    } else {
                        i in [0,active) logits[dst+i] <- panel[self+i]+b1[lane+i];
                    }
                }
            }
            lane = lane+T;
        }
        if (phase == 0) normalize();
    }
    clock_read(3);
    done[0] = 1;
    barrier(x,y);
    readout(x,y,trial);
    barrier(x,y);
    trial = trial+1;
    }
}
