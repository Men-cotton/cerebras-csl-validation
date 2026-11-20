import arch_hdr;
xp param N, MODE;
sp a[n=N] address(0x0800);
sp b[n=N] address(0x1804);
sp c[n=N] address(0x2808);
sp out[n=N] address(0x380c);
u16 counter[3] address(CE_TSC_ADDR);
u16 control[1] address(CE_PERF_CNTR_CTRL_ADDR);
u16 clocks[n=6];

function stamp(xp offset) {
    clocks[offset] = counter[0];
    clocks[offset+1] = counter[1];
    clocks[offset+2] = counter[2];
}

main {
    control[0] = control[0] | 4;
    let i in [0,N);
    stamp(0);
    if (MODE == 0) {
        ∀i out[i] <- (a[i] + b[i]) + c[i];
    } else {
        ∀i out[i] <- a[i] + b[i];
        ∀i out[i] <- out[i] + c[i];
    }
    stamp(3);
}
