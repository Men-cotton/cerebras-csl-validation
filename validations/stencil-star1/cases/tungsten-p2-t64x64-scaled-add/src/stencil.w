import arch_hdr;

socket east, west, south, north, host_in, host_out;
xp param C, R, TX, TY, E;
xp const pitch = TX+2;
xp const size = (TY+2)*pitch;
sp prev[n=size];
sp next[n=size] bank(0);
u16 col[1] address(0x7c46);
u16 row[1] address(0x7c45);
u16 tsc[3] address(CE_TSC_ADDR);
u16 control[1] address(CE_PERF_CNTR_CTRL_ADDR);
u16 clocks[n=3+6*E];
u16 xy[n=2];
u16 epochs[n=1] = {0};
// Completed sends E,W,N,S, then receives from W,E,S,N, in payload words.
u16 traffic[n=8];

function stamp(xp offset) {
    clocks[offset] = tsc[0];
    clocks[offset+1] = tsc[1];
    clocks[offset+2] = tsc[2];
}

function halo(sp active[size]) {
    let i in [1,TY+1);
    let j in [1,TX+1);
    parallel {
        { if (xy[0]+1<C) { ∀i east[] <- active[i*pitch+TX]; traffic[0] += TY; } }
        { if (xy[0]>0) { ∀i active[i*pitch] <- east[]; traffic[4] += TY; } }
    }
    parallel {
        { if (xy[0]>0) { ∀i west[] <- active[i*pitch+1]; traffic[1] += TY; } }
        { if (xy[0]+1<C) { ∀i active[i*pitch+TX+1] <- west[]; traffic[5] += TY; } }
    }
    parallel {
        { if (xy[1]>0) { ∀j north[] <- active[pitch+j]; traffic[2] += TX; } }
        { if (xy[1]+1<R) { ∀j active[(TY+1)*pitch+j] <- north[]; traffic[6] += TX; } }
    }
    parallel {
        { if (xy[1]+1<R) { ∀j south[] <- active[TY*pitch+j]; traffic[3] += TX; } }
        { if (xy[1]>0) { ∀j active[j] <- south[]; traffic[7] += TX; } }
    }
}

function jacobi(sp src[size], sp dst[size]) {
    let i in [1,TY+1);
    let j in [1,TX+1);
    ∀i ∀j dst[i*pitch+j] <- 4.0 * src[i*pitch+j];
    ∀i ∀j dst[i*pitch+j] <- dst[i*pitch+j] + src[(i-1)*pitch+j];
    ∀i ∀j dst[i*pitch+j] <- dst[i*pitch+j] + src[(i+1)*pitch+j];
    ∀i ∀j dst[i*pitch+j] <- dst[i*pitch+j] + src[i*pitch+j-1];
    ∀i ∀j dst[i*pitch+j] <- dst[i*pitch+j] + src[i*pitch+j+1];
    ∀i ∀j dst[i*pitch+j] <- 0.125 * dst[i*pitch+j];
}

xp const words = 15+6*E+2*size;

function load_host_in(xp x, xp y) {
    i in [0,size) prev[i] <- host_in[];
    c in [x+1,C) {
        i in [0,size) east[] <- (u32)host_in[];
    }
    if (x == 0) {
        r in [y+1,R) {
            c in [0,C) { i in [0,size) south[] <- (u32)host_in[]; }
        }
    }
}


function load_south(xp x, xp y) {
    i in [0,size) prev[i] <- south[];
    c in [x+1,C) {
        i in [0,size) east[] <- (u32)south[];
    }
    if (x == 0) {
        r in [y+1,R) {
            c in [0,C) { i in [0,size) south[] <- (u32)south[]; }
        }
    }
}


function load_east(xp x, xp y) {
    i in [0,size) prev[i] <- east[];
    c in [x+1,C) {
        i in [0,size) east[] <- (u32)east[];
    }
    if (x == 0) {
        r in [y+1,R) {
            c in [0,C) { i in [0,size) south[] <- (u32)east[]; }
        }
    }
}


function report_host_out(xp x, xp trial) {
    host_out[] <- (u32)trial;
    i in [0,2) host_out[] <- (u32)xy[i];
    host_out[] <- (u32)epochs[0];
    i in [0,8) host_out[] <- (u32)traffic[i];
    i in [0,3+6*E) host_out[] <- (u32)clocks[i];
    i in [0,size) host_out[] <- prev[i];
    i in [0,size) host_out[] <- next[i];
    c in [x+1,C) {
        i in [0,words) host_out[] <- (u32)west[];
    }
}


function report_south(xp x, xp trial) {
    south[] <- (u32)trial;
    i in [0,2) south[] <- (u32)xy[i];
    south[] <- (u32)epochs[0];
    i in [0,8) south[] <- (u32)traffic[i];
    i in [0,3+6*E) south[] <- (u32)clocks[i];
    i in [0,size) south[] <- prev[i];
    i in [0,size) south[] <- next[i];
    c in [x+1,C) {
        i in [0,words) south[] <- (u32)west[];
    }
}


function report_north(xp x, xp trial) {
    north[] <- (u32)trial;
    i in [0,2) north[] <- (u32)xy[i];
    north[] <- (u32)epochs[0];
    i in [0,8) north[] <- (u32)traffic[i];
    i in [0,3+6*E) north[] <- (u32)clocks[i];
    i in [0,size) north[] <- prev[i];
    i in [0,size) north[] <- next[i];
    c in [x+1,C) {
        i in [0,words) north[] <- (u32)west[];
    }
}


function report_west(xp x, xp trial) {
    west[] <- (u32)trial;
    i in [0,2) west[] <- (u32)xy[i];
    west[] <- (u32)epochs[0];
    i in [0,8) west[] <- (u32)traffic[i];
    i in [0,3+6*E) west[] <- (u32)clocks[i];
    i in [0,size) west[] <- prev[i];
    i in [0,size) west[] <- next[i];
    c in [x+1,C) {
        i in [0,words) west[] <- (u32)west[];
    }
}


// Each row gathers and broadcasts readiness, followed by each column.
// This keeps host streaming outside every measured epoch.
function fence(xp x, xp y) {
    u32 token = 1:u32;
    if (x<C-1) token <- west[];
    if (x>0) west[] <- token;
    if (x>0) token <- east[];
    if (x<C-1) east[] <- token;
    if (y<R-1) token <- north[];
    if (y>0) north[] <- token;
    if (y>0) token <- south[];
    if (y<R-1) south[] <- token;
}

function collect(xp x,xp y,xp trial) {
    if (x>0) report_west(x,trial);
    else if (y==0) report_south(x,trial);
    else if (y==1) {
        c in [0,C) { i in [0,words) host_out[] <- (u32)south[]; }
        report_host_out(x,trial);
        r in [2,R) { c in [0,C) { i in [0,words) host_out[] <- (u32)north[]; } }
    } else {
        report_north(x,trial);
        r in [y+1,R) { c in [0,C) { i in [0,words) north[] <- (u32)north[]; } }
    }
}

main {
    xp x = col[0];
    xp y = row[0];
    xy[0] = x;
    xy[1] = y;
    trial in [1,4) {
    if (x>0) load_east(x,y);
    else if (y>0) load_south(x,y);
    else load_host_in(x,y);
    i in [0,size) next[i] <- 0.0;
    i in [0,8) traffic[i] <- 0;
    control[0] = control[0] | 4;
    xp it = 0;
    fence(x,y);
    stamp(0);
    while (it<E) {
        halo(prev);
        stamp(3+6*it);
        jacobi(prev, next);
        it += 1;
        epochs[0] = it;
        if (it==E) {
            let i in [1,TY+1);
            let j in [1,TX+1);
            ∀i ∀j prev[i*pitch+j] <- next[i*pitch+j];
            ∀i next[i*pitch] <- 0.0;
            ∀i next[i*pitch+TX+1] <- 0.0;
            ∀j next[j] <- 0.0;
            ∀j next[(TY+1)*pitch+j] <- 0.0;
        }
        stamp(6*it);
        if (it<E) {
            halo(next);
            stamp(3+6*it);
            jacobi(next, prev);
            it += 1;
            epochs[0] = it;
            if (it==E) {
                let i in [1,TY+1);
                let j in [1,TX+1);
                ∀i ∀j next[i*pitch+j] <- prev[i*pitch+j];
                ∀i prev[i*pitch] <- next[i*pitch];
                ∀i prev[i*pitch+TX+1] <- next[i*pitch+TX+1];
                ∀j prev[j] <- next[j];
                ∀j prev[(TY+1)*pitch+j] <- next[(TY+1)*pitch+j];
                ∀i next[i*pitch] <- 0.0;
                ∀i next[i*pitch+TX+1] <- 0.0;
                ∀j next[j] <- 0.0;
                ∀j next[(TY+1)*pitch+j] <- 0.0;
            }
            stamp(6*it);
        }
    }
    fence(x,y);
    collect(x,y,trial);
    }
}
