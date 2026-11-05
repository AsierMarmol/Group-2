# Inference inputs

Reserved for separately supplied scoring inputs if required by the approved inference contract.
The current planned batch pipeline produces historical diagnostic predictions; it does not require live polling.
Raw-candle replay must use 60 completed consecutive candles, supported symbols, UTC times and an explicit as_of.
Demo rows may overlap training: never describe them as independent evaluation or unseen data without evidence.
Polling and API infrastructure are deferred. Input datasets are ignored by Git.
