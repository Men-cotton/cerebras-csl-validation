xp param ARM;
socket rx [[iq=0]];
socket tx;
word in [0,4) {
  u32 value=(u32)rx[];
  if (ARM==0) tx[] <- value ^ 0x61a72d90:u32;
  else tx[] <- value ^ 0x16c599af:u32;
}
