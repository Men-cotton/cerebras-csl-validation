xp param OWN;
xp param TRAILING;
socket local_rx [[iq=0]];
socket right_rx [[iq=1]];
socket tx;
xp frames=(xp)(u32)local_rx[];
frame in [0,frames) {
    xp n=(xp)(u32)local_rx[];
    i in [0,n) tx[] <- (u32)local_rx[];
    i in [0,n*(TRAILING/OWN)) tx[] <- (u32)right_rx[];
}
