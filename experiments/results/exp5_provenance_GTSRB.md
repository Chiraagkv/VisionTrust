# Exp 5 — provenance chain tampering (60-record chain, 30 trials per attack)

| attack | class | trials | detected | detection_rate |
|---|---|---|---|---|
| change output (class) | field edit | 30 | 30 | 1.000 |
| change confidence | field edit | 30 | 30 | 1.000 |
| change timestamp | field edit | 30 | 30 | 1.000 |
| change input_hash | field edit | 30 | 30 | 1.000 |
| change model_hash | field edit | 30 | 30 | 1.000 |
| change preprocessing_hash | field edit | 30 | 30 | 1.000 |
| change nonce | field edit | 30 | 30 | 1.000 |
| delete a middle record | structure | 30 | 30 | 1.000 |
| swap two adjacent records | structure | 30 | 30 | 1.000 |
| insert forged record (middle) | structure | 30 | 30 | 1.000 |
| edit + re-hash that record only | re-hash | 30 | 30 | 1.000 |
| truncate last 1–4 records | outside model | 30 | 0 | 0.000 |
| append correctly-hashed forgery | outside model | 30 | 0 | 0.000 |
| edit + re-hash all later records | outside model | 30 | 0 | 0.000 |

Cost: 61.9 ms to create a record (includes re-hashing the model file); verifying 10,000 records took 0.12 s.
