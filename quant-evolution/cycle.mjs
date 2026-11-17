import {mkdir,writeFile} from 'node:fs/promises';
import {control} from './control-client.mjs';
const out='quant-evolution/output';await mkdir(out,{recursive:true});
let result;for(let n=0;n<8;n++){result=await control({action:'CYCLE'});if(!result.backfillMayRemain)break;}
await writeFile(out+'/shadow-cycle.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
