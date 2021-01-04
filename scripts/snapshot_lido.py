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


cutoff_block = 11583452 #counting holding times and sells until that point
snapshot_block = 11543333  # as per https://mainnet.lido.fi/#/lido-dao/0x2e59a20f205bb85a89c53f1936454680651e618e/vote/16/
stETH = interface.ERC20("0xae7ab96520de3a18e5e111b5eaab095312d7fe84")
uniLP = interface.ERC20("0x4028daac072e492d34a3afdbef0ba7e35d8b55c4")
uniPair = interface.UniV2StEth("0x4028DAAC072e492d34a3Afdbef0ba7e35D8b55C4")
yveth = interface.ERC20("0x15a2B3CfaFd696e1C783FE99eed168b78a3A371e")
lido_deploy = 11473216  # https://etherscan.io/tx/0xc1ba30bd850ebd12213a61c4c2b58619ea0200a8293bace4148881fbe49cccc8
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
distributed_supply = (4/1000)*1000000000*(10**18) #0.4% of total supply

#exculde minter, uniswap and yveth vault
exclude_holders = [ZERO_ADDRESS, "0x4028daac072e492d34a3afdbef0ba7e35d8b55c4", "0x15a2B3CfaFd696e1C783FE99eed168b78a3A371e"]


def main():
    sp = staking_points()
    print("staked:", len(sp))
    hp = holding_points()
    up = lp_points()
    sellp = unisells_points() 
    yp = yveth_points()
    ysp = yveth_staking_points()
    points = merge_points([sp, hp, up, sellp, yp, ysp], exclude_holders)
    balances = compute_balances(points)
    hr_balances = human_readable_balances(balances, sp, hp, up, sellp, yp, ysp)
    distribution = prepare_merkle_tree(balances)
    save_timestamps()
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

@cached("snapshot/steth-transfers.json")
def steth_transfer_events():
    result = []
    contract = web3.eth.contract(str(stETH), abi=stETH.abi)
    step = 10000
    for start in trange(lido_deploy, cutoff_block, step):
        end = min(start + step - 1, cutoff_block)
        logs = contract.events.Transfer().getLogs(fromBlock=start, toBlock=end)
        result += [toDict(log) for log in logs]
    return result   


@cached("snapshot/unilp-transfers.json")
def unilp_transfer_events():
    result = []
    contract = web3.eth.contract(str(uniLP), abi=uniLP.abi)
    step = 10000
    for start in trange(lido_deploy, cutoff_block, step):
        end = min(start + step - 1, cutoff_block)
        logs = contract.events.Transfer().getLogs(fromBlock=start, toBlock=end)
        result += [toDict(log) for log in logs]
    return result   


@cached("snapshot/yveth-transfers.json")
def yveth_transfer_events():
    result = []
    contract = web3.eth.contract(str(yveth), abi=yveth.abi)
    step = 10000
    for start in trange(lido_deploy, cutoff_block, step):
        end = min(start + step - 1, cutoff_block)
        logs = contract.events.Transfer().getLogs(fromBlock=start, toBlock=end)
        result += [toDict(log) for log in logs]
    return result   

@cached("snapshot/unipair-swaps.json")
def unipair_swap_events():
    result = []
    contract = web3.eth.contract(str(uniPair), abi=uniPair.abi)
    step = 10000
    for start in trange(lido_deploy, cutoff_block, step):
        end = min(start + step - 1, cutoff_block)
        logs = contract.events.Swap().getLogs(fromBlock=start, toBlock=end)
        for log in logs:
            dictlog = toDict(log)
            dictlog['from'] = web3.eth.getTransaction(log['transactionHash'])['from']
            result += [dictlog]
    return result  

def transfer_events_between(logs, start_block, end_block):
    return filter(lambda log: start_block<=log["blockNumber"]<end_block, logs)

@cached("snapshot/lido-01-stake.json")
def staking_points():
    stakes = transfers_to_stakes(stETH, lido_deploy, snapshot_block)
    return stakes


@cached("snapshot/lido-02-hold.json")
def holding_points():
    points = transfers_to_holding_points(steth_transfer_events(),  
                        lido_deploy, snapshot_block, cutoff_block)
    return points


@cached("snapshot/lido-03-lp.json")
def lp_points():
    points = uniswap_lp_to_points(lido_deploy, snapshot_block, cutoff_block)
    return points

@cached("snapshot/lido-04-unisells.json")
def unisells_points():
    points = uniswap_sells_to_points()
    return points


@cached("snapshot/lido-05-yveth.json")
def yveth_points():
    points = transfers_to_holding_points(yveth_transfer_events(), 
                        lido_deploy, snapshot_block, cutoff_block)
    return points

def load_block_timestamps_cache():
    path = Path("snapshot/block-timestamps.json")
    if path.exists():
        print("load from cache", path)   
        return json.load(path.open())

@cached("snapshot/lido-05a-yveth-staking.json")
def yveth_staking_points():
    points = yveth_transfers_to_staking(yveth_transfer_events(), 
                        steth_transfer_events(),
                        lido_deploy, snapshot_block)
    return points

block_timestamps_cache = load_block_timestamps_cache()

def save_timestamps():
    path = Path("snapshot/block-timestamps.json")
    os.makedirs(path.parent, exist_ok=True)
    json.dump(block_timestamps_cache, path.open("wt"), indent=2)


@cached("snapshot/lido-06-points.json")
def merge_points(points_distributions, exclude_address):
    cnt = Counter()
    for d in points_distributions:
        cnt = cnt + Counter(d)
    points = { address: points for address, points in cnt.items()
            if points>0  and address not in exclude_address}
    sorted_points = dict(sorted(points.items(), key=lambda item: item[1]))
    return sorted_points

@cached("snapshot/lido-07-balances.json")
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

@cached("snapshot/lido-07a-human-readable-balances.json")
def human_readable_balances(balances, sp, hp, up, sellp, yp, ysp):
    hr_balances = {}
    contributions = {
        "staking": sp,
        "holding": hp,
        "lp on uniswap": up,
        "selling on uniswap": sellp,
        "yvsteth  holding": yp,
        "yvsteth staking": ysp
    }
    for holder in balances.keys():
        holder_contribuitions = {
            key: value.get(holder, 0) for
            key, value in contributions.items()
        }
        total_holder_contrubutions = sum(holder_contribuitions.values())
        annotated_contributions = {
            key: round(100*value/total_holder_contrubutions, 2)
            for key, value in holder_contribuitions.items()
        }
        hr_balances[holder] = { "balance": round(balances[holder]/10**18, 2),
            "contributions": annotated_contributions
        }
    return hr_balances




@cached("snapshot/lido-08-merkle.json")
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


def transfers_to_stakes(contract, deploy_block, snapshot_block):
    stakes = Counter()
    transfer_logs = transfer_events_between(steth_transfer_events(), deploy_block, snapshot_block)
    for log in tqdm(transfer_logs):
        if log["args"]["src"] == ZERO_ADDRESS and \
            log["args"]["dst"] != ZERO_ADDRESS:
            stakes[log["args"]["dst"]] += log["args"]["wad"]

    return dict(stakes.most_common())


def yveth_transfers_to_staking(yveth_transfer_events, steth_transfer_events, 
                        deploy_block, snapshot_block):
    points = dict()
    yv_transfer_logs = tqdm(transfer_events_between(yveth_transfer_events, deploy_block, snapshot_block))

    for yv_log in yv_transfer_logs:
        (src, dst, amount, block) =  (yv_log["args"]["src"], yv_log["args"]["dst"],
                                        yv_log["args"]["wad"], yv_log["blockNumber"])                    
        if src == ZERO_ADDRESS:
            tx_steth_transfers = filter(lambda log: log["transactionHash"] == yv_log["transactionHash"],
             steth_transfer_events)
            for st_log in tx_steth_transfers:
                if st_log["args"]["src"] == ZERO_ADDRESS:
                    points.setdefault(dst, 0)
                    points[dst] += amount
    return points
        


def seconds_between_blocks(first, second):
    (first, second) = (str(first), str(second))
    if block_timestamps_cache.get(first) is None:
        block_timestamps_cache[first] = web3.eth.getBlock(first).timestamp
    if block_timestamps_cache.get(second) is None:
        block_timestamps_cache[second] = web3.eth.getBlock(second).timestamp
     
    return block_timestamps_cache[second] - block_timestamps_cache[first]

def update_holder(holder, new_block, balance_change):
    stethwei_seconds = holder["balance"] * \
        seconds_between_blocks(holder["last_update"], new_block)
    holder["weiseconds"] += stethwei_seconds
    holder["last_update"] = new_block
    holder["balance"] += balance_change

def transfers_to_holding_points(transfer_events, deploy_block, snapshot_block, cutoff_block):
    holdings = dict()
    points = dict()
    early_transfer_logs = tqdm(transfer_events_between(transfer_events, deploy_block, snapshot_block))

    for log in early_transfer_logs:
        (src, dst, amount, block) =  (log["args"]["src"], log["args"]["dst"],
                                        log["args"]["wad"], log["blockNumber"])
        holdings.setdefault(dst, {"balance": 0, "last_update": 0, "weiseconds": 0})
                    
        if src != ZERO_ADDRESS:
            update_holder(holdings[src], block, -amount)

        update_holder(holdings[dst], block, amount)

    #after cutoff only continue to count points for original holders
    late_transfer_logs = tqdm(transfer_events_between(transfer_events, snapshot_block, cutoff_block))               
    for log in late_transfer_logs:
        (src, dst, amount, block) =  (log["args"]["src"], log["args"]["dst"],
                                        log["args"]["wad"], log["blockNumber"])

        if src != ZERO_ADDRESS and holdings.get(src):
            update_holder(holdings[src], block, -amount)
        
        if holdings.get(dst):
            update_holder(holdings[dst], block, amount)
       
    
    for holder in holdings.keys():
        update_holder(holdings[holder], cutoff_block, 0)
        points[holder] = holdings[holder]["weiseconds"]//((30*24*60*60)*2) #1 point per 2 months of holding

    return points

def uniswap_lp_to_points(deploy_block, snapshot_block, cutoff_block):
    holdings = dict()
    points = dict()
    early_transfer_logs = tqdm(transfer_events_between(unilp_transfer_events(), deploy_block, snapshot_block))

    for log in early_transfer_logs:
        (src, dst, amount, block) =  (log["args"]["src"], log["args"]["dst"],
                                        log["args"]["wad"], log["blockNumber"])
        holdings.setdefault(dst, {"balance": 0, "last_update": 0, "weiseconds": 0})
                    
        if src != ZERO_ADDRESS:
            update_holder(holdings[src], block, -amount)

        update_holder(holdings[dst], block, amount)
                   
    #after cutoff only continue to count points for original holders
    late_transfer_logs = tqdm(transfer_events_between(unilp_transfer_events(), snapshot_block, cutoff_block))
    for log in late_transfer_logs:
        (src, dst, amount, block) =  (log["args"]["src"], log["args"]["dst"],
                                        log["args"]["wad"], log["blockNumber"])

        if src != ZERO_ADDRESS and holdings.get(src):
            update_holder(holdings[src], block, -amount)
        
        if holdings.get(dst):
            update_holder(holdings[dst], block, amount)
       
    
    for holder in holdings.keys():
        update_holder(holdings[holder], cutoff_block, 0)
        points[holder] = holdings[holder]["weiseconds"]*2//((30*24*60*60)) #2 point per 1 months of holding

    return points

        

def steth_changes_per_tx(transfers, tx_hash):
    tx_steth_transfers = filter(lambda log: log["transactionHash"] == tx_hash, transfers)
    steth_balance_change = {}
    for log in tx_steth_transfers:
        (src, dst, amount) =  (log["args"]["src"], log["args"]["dst"],
                                log["args"]["wad"])
        steth_balance_change.setdefault(src, 0)
        steth_balance_change.setdefault(dst, 0)
        steth_balance_change[src] -= amount
        steth_balance_change[dst] += amount
    return steth_balance_change

def uniswap_sells_to_points():
    swaps = dict()
    points = dict()
    swap_logs = tqdm(unipair_swap_events())
    transfer_logs = steth_transfer_events()
    swap_txs = set([log["transactionHash"] for log in swap_logs])

    for tx in swap_txs:
        balance_changes =  steth_changes_per_tx(transfer_logs, tx)
        for holder in balance_changes.keys():
            swaps.setdefault(holder, {"total_sold": 0})
            swaps[holder]["total_sold"] -= balance_changes[holder]

    for seller in swaps.keys():
        if swaps[seller]["total_sold"]>0:
            points[seller] = -swaps[seller]["total_sold"]//2 #-0.5 points for every stETH sold
    return points    




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