import datetime as dt
import time
import random
import logging
import re
# Import the pipeline helper from the transformers package, which we will use to load our model
from transformers import pipeline

from optibook.synchronous_client import Exchange
from libs import print_positions_and_pnl, round_down_to_tick, round_up_to_tick

from IPython.display import clear_output

logging.getLogger('client').setLevel('ERROR')    

# Download and instantiate the facebook/bart-large-mlni model and pipeline. The first time this cell runs, it downloads a large file containing the model weights (1.5GB of parameters all in all!). 
# Let it finish and from then on it will be cached on disk.

# Stay aware that this model instantiation uses a lot of memory/RAM, as it has to load the full 1.5GB of model parameters. If you load the model as below in several notebooks at the same time, you might overload your box. 
# To avoid this, when you're done with a notebook, close it's "kernel" on the left side of the Jupyterlabs interface (The second icon. circle with a square inside, selects the active kernels). This unloads the model from memory.

c = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

INSTRUMENT_TO_NAME_DICT = {
    'CSCO' : ['Cisco', 'Cisco\'s'],
    'NVDA' : ['Nvidia', 'Nvidia\'s'],
    'ING' : ['ING', 'ING\'s'],
    'PFE' : ['Pfizer', 'Pfizer\'s'],
    'SAN': ['Santander', 'Pfizer\'s']
}
              
def insert_quotes(exchange, instrument, bid_price, ask_price, bid_volume, ask_volume):
    if bid_volume > 0:
        # Insert new bid limit order on the market
        exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=bid_price,
            volume=bid_volume,
            side='bid',
            order_type='limit',
        )
        
        # Wait for some time to avoid breaching the exchange frequency limit
        time.sleep(0.05)

    if ask_volume > 0:
        # Insert new ask limit order on the market
        exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=ask_price,
            volume=ask_volume,
            side='ask',
            order_type='limit',
        )

        # Wait for some time to avoid breaching the exchange frequency limit
        time.sleep(0.05)
        
def insert_ioc(exchange, instrument, bid_price, ask_price, position):
    if position < 0:
        # Insert new bid ioc order on the market
        exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=ask_price,
            volume=-position,
            side='bid',
            order_type='ioc',
        )
        
        # Wait for some time to avoid breaching the exchange frequency limit
        time.sleep(0.05)

    if position > 0:
        # Insert new ask ioc order on the market
        exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=bid_price,
            volume=position,
            side='ask',
            order_type='ioc',
        )

        # Wait for some time to avoid breaching the exchange frequency limit
        time.sleep(0.05)
        
def analyse_feeds(feeds):
    if not feeds:
        return True
    for feed in feeds:
        if mean(c(feed.post, ['Risky', 'Worried', 'Scared', 'Problematic'])['scores']) > 0.5:
            return True
    return False

def get_risky_feeds(feeds):
    risky = []
    if not feeds:
        return risky
    for feed in feeds:
        if any(score > 0.4 for score in c(feed.post, ['Risky', 'Worried', 'Scared', 'Problematic'])['scores']):
            risky.append(feed)
    return risky

def get_optimistic_feeds(feeds):
    optimistic = []
    if not feeds:
        return optimistic
    for feed in feeds:
        if c(feed.post, ['Optimistic'])['scores'][0] > 0.5:
            optimistic.append(feed)
    return optimistic

def is_related(feeds, instrument_id):
    for feed in feeds:
              for name in INSTRUMENT_TO_NAME_DICT[instrument_id]:
                  if re.compile(r'\b({0})\b'.format(name), flags=re.IGNORECASE).search(feed.post):
                        return True
    return False

exchange = Exchange()
exchange.connect()

INSTRUMENTS = exchange.get_instruments()

QUOTED_VOLUME = 10
FIXED_MINIMUM_CREDIT = 0.15
PRICE_RETREAT_PER_LOT = 0.005
POSITION_LIMIT = 100


while True:
    social_feeds = exchange.poll_new_social_media_feeds()
    # is_risky = analyse_feeds(social_feeds)
    risky_feeds = get_risky_feeds(social_feeds)
    optimistic_feeds = get_optimistic_feeds(social_feeds)
    
    for instrument in INSTRUMENTS.values():
        # Remove all existing (still) outstanding limit orders
        exchange.delete_orders(instrument.instrument_id)
    
        # Obtain order book and only skip this instrument if there are no bids or offers available at all on that instrument,
        # as we we want to use the mid price to determine our own quoted price
        instrument_order_book = exchange.get_last_price_book(instrument.instrument_id)
        if not (instrument_order_book and instrument_order_book.bids and instrument_order_book.asks):
            continue
    
        # Obtain own current position in instrument
        position = exchange.get_positions()[instrument.instrument_id]

        # Obtain best bid and ask prices from order book to determine mid price
        best_bid_price = instrument_order_book.bids[0].price
        best_ask_price = instrument_order_book.asks[0].price
        mid_price = (best_bid_price + best_ask_price) / 2.0 
        
        # Calculate our fair/theoretical price based on the market mid price and our current position
        theoretical_price = mid_price - PRICE_RETREAT_PER_LOT * position

        # Calculate final bid and ask prices to insert
        bid_price = round_down_to_tick(theoretical_price - FIXED_MINIMUM_CREDIT, instrument.tick_size)
        ask_price = round_up_to_tick(theoretical_price + FIXED_MINIMUM_CREDIT, instrument.tick_size)
        
        if bid_price < best_ask_price and ask_price > best_bid_price and best_bid_price > best_ask_price:
            bid_price = best_ask_price
            ask_price = best_bid_price
            
            
        
        # Calculate bid and ask volumes to insert, taking into account the exchange position_limit
        max_volume_to_buy = POSITION_LIMIT - position
        max_volume_to_sell = POSITION_LIMIT + position

        bid_volume = min(QUOTED_VOLUME, max_volume_to_buy)
        ask_volume = min(QUOTED_VOLUME, max_volume_to_sell)
        
        if is_related(risky_feeds, instrument.instrument_id):
            # exchange.delete_orders(instrument.instrument_id)
            insert_ioc(exchange, instrument, best_bid_price, best_ask_price, position)
            time.sleep(10);
        elif is_related(optimistic_feeds, instrument.instrument_id):
            exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=best_ask_price,
            volume=bid_volume,
            side='bid',
            order_type='ioc',
            )
        else:
            insert_quotes(exchange, instrument, bid_price, ask_price, bid_volume, ask_volume)
            
            
    # Wait for a few seconds to refresh the quotes
    time.sleep(2)
    
    # Clear the displayed information after waiting
    clear_output(wait=True)
    
