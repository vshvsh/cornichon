# Lido Airdrop Merkle distributoe

This repo contains everything to produce a snapshot of airdrop balances for Lido airdrop.


## Validation

To generate the snapshot data:

```
pip install -r requirements.txt

brownie networks add Ethereum mainnet host=$YOUR_MAINNET_NODE chainid=1

rm -rf snapshot
brownie run snapshot_lido --network mainnet
```
