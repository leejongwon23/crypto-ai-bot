import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {hash,promotionGate} from './lib.mjs';
import {POLICY_TEXT,POLICY_SHA256} from '../v17-common/policy-v19.mjs';
import {evaluateRangeStopContext} from '../v17-common/regime-entry-risk.mjs';
const dir=process.argv[2]??'research-output';await mkdir(dir,{recursive:true});
const source=await readFile('research-v2/incident-source-fixture.txt','utf8');
const original={id:'V19-ETH-SHORT-20261117-063519-RANGE1826',symbol:'ETHUSDT',direction:'SHORT',entryTactic:'RANGE_BOUNDARY',entryLow:1823.63,entryHigh:1823.63,stop:1828.91,tp1:1819.37,tp2:1811.19,tp3:1809.22};
const examples=[
 {name:'ONE_REAL_LEDGER_CASE_REDUCED_SOURCE_REPLAY',signal:original,raw:source,synthetic:false},
 {name:'SYNTHETIC_EQUAL_BOUNDARY',signal:{...original,stop:1836.88},raw:source,synthetic:true},
 {name:'SYNTHETIC_WIDER_STOP',signal:{...original,stop:1839.5},raw:source,synthetic:true},
 {name:'SYNTHETIC_OPPOSITE_ALIGN',signal:original,raw:source.replaceAll('align=BULL','align=BEAR'),synthetic:true},
 {name:'SYNTHETIC_TREND_TACTIC',signal:{...original,entryTactic:'TREND_CONTINUATION'},raw:source,synthetic:true}
].map(x=>({name:x.name,synthetic:x.synthetic,decision:evaluateRangeStopContext({raw:x.raw,signal:x.signal}),rr2:(x.signal.entryLow-x.signal.tp2)/(x.signal.stop-x.signal.entryLow)}));
await writeFile(join(dir,'gate-replay.json'),JSON.stringify({examples,limitation:'Reduced fixture is from existing code test and ledger fields; not a complete incident-time raw snapshot. No new policy effect or realized performance is measured.'},null,2)+'\n');
if(hash(POLICY_TEXT)!==POLICY_SHA256)throw Error('BASELINE_POLICY_MISMATCH');
await writeFile(join(dir,'MASTER_PROMPT_V1.9_FINAL_original.txt'),POLICY_TEXT);
const protocol=JSON.parse(await readFile('research-v2/protocol.draft.json','utf8'));
await writeFile(join(dir,'promotion-status.json'),JSON.stringify(promotionGate(protocol,{}),null,2)+'\n');
const paths=['v17-common/server.mjs','v17-common/v19-core.mjs','v17-common/regime-entry-risk.mjs','v17-common/signal-table-view.mjs','research-v2/lib.mjs','research-v2/collect.mjs'];
const manifest=[];for(const path of paths){const content=await readFile(path,'utf8');manifest.push({path,sha256:hash(content),bytes:Buffer.byteLength(content)});}
await writeFile(join(dir,'candidate-file-hashes.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({policySha256:POLICY_SHA256,examples:examples.length,realCases:1,promotion:'BLOCKED',signalGenerationModelChanged:false,priceTrackingChanged:true}));
