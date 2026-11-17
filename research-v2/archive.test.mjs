import test from 'node:test';
import assert from 'node:assert/strict';
import {aggregateCSV} from './archive.mjs';
const header='timestamp,symbol,side,size,price,trdMatchID\n';
test('public raw trades produce observed minute OHLC and signed flow without missing minute interpolation',()=>{
 const x=aggregateCSV(header+'60,ETHUSDT,Buy,2,100,a\n61,ETHUSDT,Sell,3,102,b\n180,ETHUSDT,Sell,1,99,c','ETHUSDT');
 assert.equal(x.tradeCount,3);assert.equal(x.bars.length,2);assert.equal(x.bars[0].cvdDelta,-1);assert.equal(x.bars[0].high,102);
 assert.equal(x.bars[1].start,180000);
});
test('wrong symbol, reversed time and duplicate IDs reject',()=>{
 assert.throws(()=>aggregateCSV(header+'60,BTCUSDT,Buy,2,100,a','ETHUSDT'));
 assert.throws(()=>aggregateCSV(header+'61,ETHUSDT,Buy,2,100,a\n60,ETHUSDT,Sell,3,102,b','ETHUSDT'));
 assert.throws(()=>aggregateCSV(header+'60,ETHUSDT,Buy,2,100,a\n61,ETHUSDT,Sell,3,102,a','ETHUSDT'));
});
