# frb_assembler
A lightweight tcp packet assembler for real-time FRB detection

Suggested companion repositories:
  dedisp-contain (trigger container object)
  trigger_handler (trigger handling suite)
  mpl_interface (plots with memory leak, no extra charge)
  kotekan (L0 baseband cross correlator)

currently this software is geared towards use with the following downstream dedisperse:
  bonsai (proprietary CHIMEFRB dedisperer)
  rf_pipelines (also designed for CHIMEFRB)

but in actuality it can be easily modified to work with any dedispersion/detection scheme
similarly, you can accept data from any conforming tcp stream.
