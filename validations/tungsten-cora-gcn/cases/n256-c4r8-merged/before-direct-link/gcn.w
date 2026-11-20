// Two-layer E236 GCN with column-local endpoint delivery.
// Row reduction and broadcast assemble the projected features. Each remote
// destination receives the distinct source vectors used by its edge list.
// Contiguous destination slots can share a packet; source group and occurrence
// order are preserved when the destination accumulates its edges.
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
xp param NRX;
xp param NSEND;
xp param NPLAN;
xp param ENV;
xp param TRIM;
socket mp;
xp param ITER;
xp param X0 = 0;
xp param Y0 = 0;

socket rr;                  // receive right, send left
socket rb;                  // left-root row broadcast
socket yp;                  // send to y+1, receive from y-1
socket ym;                  // send to y-1, receive from y+1

u16 x_ptr[n=K+1];
u16 x_col[n=MX];
sp x_val[n=MX];
sp w0[n=FW*H];
sp w1[n=H*O];
u16 edge_ptr[n=L+2];
u16 split_control[n=2];
u32 split_words[n=2];
u16 edge_src[n=MG];
sp edge_weight[n=MG];
u16 slots[n=L];
u16 counts[n=6];
u16 send_slot[n=NSEND];
u32 packet_header[n=2*NPLAN];
u32 packet_tag[n=2*NPLAN];
u16 packet_offset[n=2*NPLAN];
u16 packet_count[n=2*NPLAN];
u32 envelope[n=ENV];
u32 traffic[n=4];
sp pre[n=L*H];
sp logits[n=L*O];
sp panel_local[n=K*T];
sp hidden_tail[n=K*T];
sp panel_sum[n=(K+NRX)*T];
sp accum[n=(L+1)*T];
u32 stats[n=8];
u32 completion[n=1] = {0};
u32 epochs[n=ITER];
sp pre_history[n=ITER*L*H];
sp logits_history[n=ITER*L*O];
sp panel_history[n=ITER*K*T];
u32 stats_history[n=ITER*8];
u16 clocks[n=ITER*6];

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

function row_collective(xp x) {
    if (x == C-1) {
        i in [0, K*T) rr[] <- panel_local[i];
    } else if (x > 0) {
        i in [0, K*T) rr[] <- rr[] + panel_local[i];
    } else {
        i in [0, K*T) panel_sum[i] <- rr[] + panel_local[i];
    }
    stats[1] = stats[1] + 1;
    if (x == 0) {
        i in [0, K*T) rb[] <- panel_sum[i];
    } else {
        i in [0, K*T) panel_sum[i] <- rb[];
    }
    stats[2] = stats[2] + 1;
}

// The first K slots retain the local row result; remote vectors follow.
function exchange(xp x, xp y, xp active) {
    xp width = T;
    xp stage = 0;
    if (active < T) {
        stage = 1;
        if (TRIM == 1) width = active;
    }
    xp sent = counts[stage];
    xp received = counts[2+stage];
    xp total = counts[4+stage];
    xp cursor = 0;
    j in [0,sent) {
        xp at = stage*NPLAN+j;
        xp offset = packet_offset[at];
        xp num = packet_count[at];
        envelope[cursor] = packet_header[at];
        envelope[cursor+1] = packet_tag[at];
        cursor = cursor+2;
        v in [0,num) {
            xp src = send_slot[offset+v]*T;
            i in [0,width) envelope[cursor+i] <- bitcast(u32,panel_sum[src+i]);
            cursor = cursor+width;
        }
    }
    barrier(x,y);
    parallel {
        { if (sent > 0) {
            i in [0,total) mp[] <- envelope[i];
        } }
        { if (received > 0) {
            j in [0,received) {
                u32 header;
                u32 tag;
                header <- mp[];
                tag <- mp[];
                xp dst = (K+(xp)(tag & 0xffff))*T;
                xp num = (xp)(tag >> 16);
                v in [0,num) {
                    i in [0,width) panel_sum[dst+i] <- mp[];
                    dst = dst+T;
                }
                traffic[3] = traffic[3]+(u32)(2+num*width);
            }
        } }
    }
    traffic[0] = traffic[0]+(u32)sent;
    traffic[1] = traffic[1]+(u32)received;
    traffic[2] = traffic[2]+(u32)cursor;
    stats[3] = stats[3]+(u32)received;
}

main {
    // Normalize the physical application origin before any measured interval.
    xp px = col_id[0] - X0;
    xp py = row_id[0] - Y0;
    tsc_control[0] = tsc_control[0] | 4;
    i in [0,4) traffic[i] <- 0;
    i in [0,2) split_words[i] <- 0;
    xp iteration = 0;
    while (iteration < ITER) {
        i in [0, 8) stats[i] <- 0;
        i in [0, L*H) pre[i] <- 0.0;
        i in [0, L*O) logits[i] <- 0.0;
        barrier(px, py);
        xp phase = 0;
        xp lane = 0;
        clock_read(iteration*6);
        stats[4] = (u32)(x_ptr[K]*H);
        while (phase < 2) {
            xp width = H;
            if (phase == 1) width = O;
            while (lane < width) {
                xp active = T;
                if (lane + T > width) active = width - lane;
                if (phase == 1) i in [0, K*T) panel_local[i] <- 0.0;
                i in [0, K*T) panel_sum[i] <- 0.0;
                i in [0, L*T) accum[i] <- 0.0;
                if (split_control[0] == 2) i in [0,T) accum[L*T+i] <- 0.0;
                if (phase == 0) {
                    if (lane == 0) {
                        vertex in [0,K) {
                            xp e = x_ptr[vertex];
                            xp stop = x_ptr[vertex+1];
                            xp dst = vertex*T;
                            i in [0,T) panel_local[dst+i] <- 0.0;
                            i in [0,T) hidden_tail[dst+i] <- 0.0;
                            while (e < stop) {
                                xp w = x_col[e]*H;
                                sp val = x_val[e];
                                i in [0,T) panel_local[dst+i] <+- w0[w+i]*val;
                                i in [0,T) hidden_tail[dst+i] <+- w0[w+T+i]*val;
                                e = e+1;
                            }
                        }
                    } else {
                        i in [0,K*T) panel_local[i] <- hidden_tail[i];
                    }
                } else {
                    xp v = 0;
                    while (v < L) {
                        if (slots[v] != 65535) {
                            xp h = 0;
                            while (h < H) {
                                sp value = pre[v*H+h];
                                if (!(value > 0.0)) value = 0.0;
                                xp dst_base = slots[v]*T;
                                xp src_base = h*O+lane;
                                i in [0, active) panel_local[dst_base+i] <+- w1[src_base+i] * value;
                                stats[6] = stats[6] + (u32)active;
                                h = h + 1;
                            }
                        }
                        v = v + 1;
                    }
                }
                row_collective(px);
                exchange(px,py,active);
                xp vertices = L;
                if (split_control[0] == 2) vertices = L+1;
                vertex in [0,vertices) {
                    xp dst_base = vertex*T;
                    xp e = edge_ptr[vertex];
                    xp stop = edge_ptr[vertex+1];
                    while (e < stop) {
                        xp src_base = edge_src[e]*T;
                        sp weight = edge_weight[e];
                        i in [0,active) accum[dst_base+i] <+- panel_sum[src_base+i]*weight;
                        stats[5] = stats[5] + (u32)active;
                        e = e+1;
                    }
                }
                if (split_control[0] == 2) {
                    i in [0,active) rr[] <- accum[L*T+i];
                    split_words[0] = split_words[0]+(u32)active;
                } else if (split_control[0] == 3) {
                    i in [0,active) rr[] <- rr[];
                    split_words[0] = split_words[0]+(u32)active;
                    split_words[1] = split_words[1]+(u32)active;
                } else if (split_control[0] == 1) {
                    xp slot = (xp)split_control[1]*T;
                    i in [0,active) accum[slot+i] <+- rr[];
                    split_words[1] = split_words[1]+(u32)active;
                }
                xp v = 0;
                while (v < L) {
                    if (phase == 0) {
                        i in [0, active) pre[v*H+lane+i] <- accum[v*T+i];
                    } else {
                        i in [0, active) logits[v*O+lane+i] <- accum[v*T+i];
                    }
                    v = v + 1;
                }
                stats[7] = stats[7] + 1;
                lane = lane + T;
            }
            phase = phase + 1;
            lane = 0;
        }
        clock_read(iteration*6+3);
        stats[0] = 0x600d;
        i in [0, L*H) pre_history[iteration*L*H+i] <- pre[i];
        i in [0, L*O) logits_history[iteration*L*O+i] <- logits[i];
        i in [0, K*T) panel_history[iteration*K*T+i] <- panel_sum[i];
        i in [0, 8) stats_history[iteration*8+i] <- stats[i];
        completion[0] = completion[0] + 1;
        epochs[iteration] = completion[0];
        barrier(px, py);
        iteration = iteration + 1;
    }
}
