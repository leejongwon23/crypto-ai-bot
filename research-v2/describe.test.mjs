import test from 'node:test';
import assert from 'node:assert/strict';
import {aggregateMinutes} from './describe.mjs';
const b=(i)=>({start:i*60000,open:100+i,close:101+i,high:102+i,low:99+i,volume:1,turnover:100+i,cvdDelta:i%2?1:-1});
test('derived OHLCV uses complete observed intervals only',()=>{
 const r=aggregateMinutes([b(0),b(1),b(2),b(3),b(4)],5);
 assert.equal(r.bars.length,1);assert.equal(r.bars[0].volume,5);assert.equal(r.bars[0].open,100);assert.equal(r.bars[0].close,105);
 const gap=aggregateMinutes([b(0),b(1),b(3),b(4)],5);
 assert.equal(gap.bars.length,0);assert.equal(gap.omittedIncompleteBuckets.length,1);
});
