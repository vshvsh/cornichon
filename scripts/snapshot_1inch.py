import json
import os
from collections import Counter
from fractions import Fraction
from functools import wraps
from itertools import zip_longest
from pathlib import Path

from brownie import Wei, accounts, chain, interface, web3
from eth_abi.packed import encode_abi_packed
from eth_utils import encode_hex
from tqdm import trange, tqdm
from hexbytes import HexBytes

from joblib import Memory
cachedir = 'oneinch_cache'
mem = Memory(cachedir)

snapshot_block = 11953268  # now
LDO = interface.ERC20("0x5a98fcbea516cf06857215779fd812ca3bef1b32")
inch = interface.ERC20("0x111111111117dc0aa78b770fa6a738034120c302")
sr_1inch = interface.FarmingRewards("0x8Acdb3bcC5101b1Ba8a5070F003a77A2da376fe8")


sr_1inch_deploy = 11771178  # https://etherscan.io/tx/0xc1ba30bd850ebd12213a61c4c2b58619ea0200a8293bace4148881fbe49cccc8
distributed_supply = 250_000*(10**18) #250k tokens



def main():
    tp = transfer_points()
    print("claimed:", len(tp))
    rp = remaining_points()
    points = merge_points([tp, rp], [])
    points_sum = sum_points(points)*1.0/10**18
    print(f"total 1inch tokens {points_sum}")
    balances = compute_balances(points)
    distribution = prepare_merkle_tree(balances)
    print("recipients:", len(balances))
    print("total supply:", sum(balances.values()) / 1e18)
    print("merkle root:", distribution["merkleRoot"])


def cached(path):
    path = Path(path)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if path.exists():
                print("load from cache", path)
                return json.load(path.open())
            else:
                result = func(*args, **kwargs)
                if result is None:
                    return
                os.makedirs(path.parent, exist_ok=True)
                json.dump(result, path.open("wt"), indent=2)
                print("write to cache", path)
                return result

        return wrapper

    return decorator



def toDict(dictToParse):
    # convert any 'AttributeDict' type found to 'dict'
    parsedDict = dict(dictToParse)
    for key, val in parsedDict.items():
        # check for nested dict structures to iterate through
        if  'dict' in str(type(val)).lower():
            parsedDict[key] = toDict(val)
        # convert 'HexBytes' type to 'str'
        elif 'HexBytes' in str(type(val)):
            parsedDict[key] = val.hex()
    return parsedDict

def oneinch_reward_events():
    result = []
    contract = web3.eth.contract(str(sr_1inch), abi=sr_1inch.abi)
    step = 10000
    for start in trange(sr_1inch_deploy, snapshot_block, step):
        end = min(start + step - 1, snapshot_block)
        logs = contract.events.RewardPaid().getLogs(fromBlock=start, toBlock=end)
        result += [toDict(log) for log in logs]
    return result   

oneinch_reward_events_mem = mem.cache(oneinch_reward_events)

def stake_events():
    result = []
    contract = web3.eth.contract(str(sr_1inch), abi=sr_1inch.abi)
    step = 10000
    for start in trange(sr_1inch_deploy, snapshot_block, step):
        end = min(start + step - 1, snapshot_block)
        logs = contract.events.Staked().getLogs(fromBlock=start, toBlock=end)
        result += [toDict(log) for log in logs]
    return result   

stake_events_mem = mem.cache(stake_events)

def rewards_to_points():
    stakes = Counter()
    transfer_logs = oneinch_reward_events_mem() 
    for log in tqdm(transfer_logs):
        stakes[log["args"]["user"]] += log["args"]["reward"]

    return dict(stakes.most_common())


@cached("snapshot/oneinch-01-transfer.json")
def transfer_points():
    points = rewards_to_points()
    return points


@cached("snapshot/oneinch-02-remaining.json")
def remaining_points():

    stakers = {} 
    for log in tqdm(stake_events_mem()):
        stakers[log["args"]["user"]] = 0
    
    for staker in tqdm(stakers.keys()):
        stakers[staker] = sr_1inch.earned(staker, block_identifier=11953269)

    return stakers


@cached("snapshot/oneinch-03-points.json")
def merge_points(points_distributions, exclude_address):
    cnt = Counter()
    for d in points_distributions:
        cnt = cnt + Counter(d)
    points = { address: points for address, points in cnt.items()
            if points>0  and address not in exclude_address}
    sorted_points = dict(sorted(points.items(), key=lambda item: item[1]))
    return sorted_points

@cached("snapshot/oneinch-04-balances.json")
def compute_balances(points_per_address):
    total_ldo = distributed_supply
    left_ldo = total_ldo
    total_points = sum(points_per_address.values())
    left_points = total_points
    balances = {}
    for address in points_per_address.keys():
        addr_ldo = int(left_ldo * Fraction(points_per_address[address], left_points))
        left_points -= points_per_address[address]
        balances[address] = addr_ldo
        left_ldo -= addr_ldo
    return balances


def sum_points(points):
    return sum(points.values())

@cached("snapshot/oneinch-05-merkle.json")
def prepare_merkle_tree(balances):
    elements = [
        (index, account, amount)
        for index, (account, amount) in enumerate(balances.items())
    ]
    nodes = [
        encode_hex(encode_abi_packed(["uint", "address", "uint"], el))
        for el in elements
    ]
    tree = MerkleTree(nodes)
    distribution = {
        "merkleRoot": encode_hex(tree.root),
        "tokenTotal": hex(sum(balances.values())),
        "claims": {
            user: {
                "index": index,
                "amount": hex(amount),
                "proof": tree.get_proof(nodes[index]),
            }
            for index, user, amount in elements
        },
    }
    return distribution



class MerkleTree:
    def __init__(self, elements):
        self.elements = sorted(set(web3.keccak(hexstr=el) for el in elements))
        self.layers = MerkleTree.get_layers(self.elements)

    @property
    def root(self):
        return self.layers[-1][0]

    def get_proof(self, el):
        el = web3.keccak(hexstr=el)
        idx = self.elements.index(el)
        proof = []
        for layer in self.layers:
            pair_idx = idx + 1 if idx % 2 == 0 else idx - 1
            if pair_idx < len(layer):
                proof.append(encode_hex(layer[pair_idx]))
            idx //= 2
        return proof

    @staticmethod
    def get_layers(elements):
        layers = [elements]
        while len(layers[-1]) > 1:
            layers.append(MerkleTree.get_next_layer(layers[-1]))
        return layers

    @staticmethod
    def get_next_layer(elements):
        return [
            MerkleTree.combined_hash(a, b)
            for a, b in zip_longest(elements[::2], elements[1::2])
        ]

    @staticmethod
    def combined_hash(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return web3.keccak(b"".join(sorted([a, b])))