import test from 'node:test';
import assert from 'node:assert/strict';
process.env.V17_NO_LISTEN='1';
delete process.env.DATABASE_URL;
const {mcp,tools}=await import('./server.mjs');
async function call(name,scope='signals:read signals:write'){
 const body=JSON.stringify({jsonrpc:'2.0',id:1,method:'tools/call',params:{name,arguments:{}}});
 const req={async *[Symbol.asyncIterator](){yield Buffer.from(body);}};
 let result;
 const res={writeHead(){},end(x){result=JSON.parse(x);}};
 await mcp(req,res,{scope});return result;
}
for(const name of ['publish_official_v17_analysis','publish_official_v18_analysis','publish_official_v19_analysis']){
 test(name+' is listed and reaches the same publication handler',async()=>{
  assert.ok(tools.some(t=>t.name===name));
  const r=await call(name);
  assert.equal(r.error,undefined);
  assert.equal(r.result.isError,true);
  assert.equal(r.result.content[0].text,'PERSISTENT_DATABASE_REQUIRED');
 });
 test(name+' requires write scope before dispatch',async()=>{
  const r=await call(name,'signals:read');
  assert.equal(r.result.content[0].text,'OAUTH_SCOPE_SIGNALS_WRITE_REQUIRED');
 });
}
test('an unlisted method stays unknown',async()=>assert.equal((await call('not_a_real_tool')).error.code,-32601));
