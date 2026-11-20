import arch_hdr;
xp param N;
socket rx [[iq=0]];
socket tx;
sp a[n=N] address(0x0800);
sp b[n=N] address(0x1804);
sp c[n=N] address(0x2808);
sp out[n=N] address(0x380c);
u16 counter[3] address(CE_TSC_ADDR);
u16 control[1] address(CE_PERF_CNTR_CTRL_ADDR);
u16 clocks[n=6];

function stamp(xp offset) {
    u16 high = counter[2];
    u16 middle = counter[1];
    u16 low = counter[0];
    u16 middle_after = counter[1];
    u16 high_after = counter[2];
    while ((high != high_after) || (middle != middle_after)) {
        high = high_after;
        middle = counter[1];
        low = counter[0];
        middle_after = counter[1];
        high_after = counter[2];
    }
    clocks[offset] = low;
    clocks[offset+1] = middle;
    clocks[offset+2] = high;
}

main {
    control[0] = control[0] | 4;
    let i in [0,N);
    block in [0,4) {
        u32 tag = (u32)rx[];
        u32 serial = (u32)rx[];
        u32 mode = (u32)rx[];
        j in [0,N) a[j] <- rx[];
        j in [0,N) b[j] <- rx[];
        j in [0,N) c[j] <- rx[];
        turn in [0,4) {
            if (mode == 0:u32) {
                stamp(0);
                ∀i out[i] <- (a[i] + b[i]) + c[i];
                stamp(3);
            } else {
                stamp(0);
                ∀i out[i] <- a[i] + b[i];
        ∀i out[i] <- out[i] + c[i];
                stamp(3);
            }
            tx[] <- tag;
            tx[] <- serial;
            tx[] <- (u32)turn;
            tx[] <- mode;
            j in [0,6) tx[] <- (u32)clocks[j];
            j in [0,N) tx[] <- out[j];
            mode = 1:u32-mode;
        }
    }
}
