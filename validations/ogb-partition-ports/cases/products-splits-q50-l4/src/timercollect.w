// Gather owner frames from right to left, preserving x order.
xp param OWN;
xp param TRAILING;
socket local_rx [[iq=0]];
socket right_rx [[iq=1]];
socket tx;
sample in [0,5) { i in [0,10) tx[] <- (u32)local_rx[]; }
frame in [0,12) {
    i in [0,2) tx[] <- (u32)local_rx[];
    i in [0,OWN) tx[] <- (u32)local_rx[];
    i in [0,TRAILING) tx[] <- (u32)right_rx[];
    i in [0,8) tx[] <- (u32)local_rx[];
}

sample in [0,5) { i in [0,10) tx[] <- (u32)local_rx[]; }
