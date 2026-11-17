import {readFile,writeFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {sha} from '../v17-common/quant-structure.mjs';
const proof={sourceCommit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),workflowRun:process.env.GITHUB_RUN_ID,kind:'ACTUAL_CI_TEST_OUTPUT'};
for(const [kind,path]of [['unit','quant-evolution/output/unit.tap'],['integration','quant-evolution/output/integration.tap']]){
 const bytes=await readFile(path,'utf8');if(!/^# fail 0\s*$/m.test(bytes)||!/^# skipped 0\s*$/m.test(bytes)||!/^# tests [1-9]\d*\s*$/m.test(bytes))throw Error('REAL_CI_TEST_OUTPUT_NOT_PASSING');proof[kind]={path,sha256:sha(bytes)};
}
await writeFile('quant-evolution/output/ci-proof-manifest.json',JSON.stringify(proof,null,2)+'\n');
