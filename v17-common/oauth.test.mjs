import test from 'node:test';import assert from 'node:assert/strict';import http from 'node:http';import crypto from 'node:crypto';import {makeOAuth} from './oauth.mjs';
const origin='https://btc-eth-v17-canonical-api.onrender.com';
function db(){const clients=new Map(),codes=new Map(),tokens=new Map(),refresh=new Map();
 const query=async(sql,args=[])=>{
  if(sql.includes('CREATE TABLE')||sql==='BEGIN'||sql==='COMMIT'||sql==='ROLLBACK')return {rows:[]};
  if(sql.startsWith('INSERT INTO v17_oauth_clients')){clients.set(args[0],JSON.parse(args[1]));return {rows:[]};}
  if(sql.startsWith('SELECT client_id,redirect_uris'))return {rows:clients.has(args[0])?[{client_id:args[0],redirect_uris:clients.get(args[0])}]:[]};
  if(sql.startsWith('INSERT INTO v17_oauth_codes')){codes.set(args[0],{client_id:args[1],redirect_uri:args[2],code_challenge:args[3],scope:args[4],expires_at:new Date(Date.now()+300000),used:false});return {rows:[]};}
  if(sql.startsWith('SELECT client_id,redirect_uri,code_challenge'))return {rows:codes.has(args[0])?[codes.get(args[0])]:[]};
  if(sql.startsWith('UPDATE v17_oauth_codes')){codes.get(args[0]).used=true;return {rows:[]};}
  if(sql.startsWith('INSERT INTO v17_oauth_access')){tokens.set(args[0],{client_id:args[1],scope:args[2]});return {rows:[]};}
  if(sql.startsWith('INSERT INTO v17_oauth_refresh')){refresh.set(args[0],{client_id:args[1],scope:args[2],used:false,expires_at:new Date(Date.now()+2592000000)});return {rows:[]};}
  if(sql.startsWith('SELECT client_id,scope FROM v17_oauth_access'))return {rows:tokens.has(args[0])?[tokens.get(args[0])]:[]};
  if(sql.startsWith('SELECT client_id,scope,used,expires_at FROM v17_oauth_refresh'))return {rows:refresh.has(args[0])?[refresh.get(args[0])]:[]};
  if(sql.startsWith('UPDATE v17_oauth_refresh')){refresh.get(args[0]).used=true;return {rows:[]};}
  throw Error('unimplemented mock query '+sql);
 };return {query,connect:async()=>({query,release(){}})};
}
function setup(){const oauth=makeOAuth({origin,pool:db(),ownerSecret:'THIS_IS_TEST_SECRET_1234'});const server=http.createServer(async(req,res)=>{try{const handled=await oauth.handle(req,res);if(handled===false){res.writeHead(404);res.end();}}catch(e){res.writeHead(500);res.end(e.message)}});return {oauth,server};}
function post(base,path,b,kind='json'){const headers={'Content-Type':kind==='json'?'application/json':'application/x-www-form-urlencoded'};return fetch(base+path,{method:'POST',headers,body:kind==='json'?JSON.stringify(b):new URLSearchParams(b),redirect:'manual'});}
test('OAuth discovery, DCR, PKCE S256, login, code reuse, refresh and audience isolation',async()=>{
 const {oauth,server}=setup();await oauth.init();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const base=`http://127.0.0.1:${server.address().port}`;
 try{
  const md=await (await fetch(base+'/.well-known/oauth-protected-resource')).json();assert.equal(md.resource,origin+'/mcp');
  const as=await (await fetch(base+'/.well-known/oauth-authorization-server')).json();assert(as.code_challenge_methods_supported.includes('S256'));
  let res=await post(base,'/oauth/register',{redirect_uris:['https://evil.example/redirect']});assert.equal(res.status,400);
  const redirect='https://chatgpt.com/connector/oauth/123xyz';res=await post(base,'/oauth/register',{redirect_uris:[redirect],token_endpoint_auth_method:'none'});assert.equal(res.status,201);const {client_id}=await res.json();assert(client_id.startsWith('v17_'));
  const verifier='a'.repeat(43),challenge=crypto.createHash('sha256').update(verifier).digest('base64url');const params={response_type:'code',client_id,redirect_uri:redirect,code_challenge:challenge,code_challenge_method:'S256',resource:origin+'/mcp',scope:'signals:read signals:write',state:'x'};
  res=await fetch(base+'/oauth/authorize?'+new URLSearchParams(params));assert.equal(res.status,200);assert((await res.text()).includes('SIGNAL_API_KEY'));
  res=await post(base,'/oauth/authorize',{...params,password:'bad'},'form');assert.equal(res.status,401);
  res=await post(base,'/oauth/authorize',{...params,password:'THIS_IS_TEST_SECRET_1234'},'form');assert.equal(res.status,302);const to=new URL(res.headers.get('location'));assert.equal(to.origin,'https://chatgpt.com');assert.equal(to.searchParams.get('state'),'x');const code=to.searchParams.get('code');assert(code);
  let fields={grant_type:'authorization_code',client_id,code,code_verifier:'b'.repeat(43),redirect_uri:redirect,resource:origin+'/mcp'};
  res=await post(base,'/oauth/token',fields,'form');assert.equal(res.status,400);
  fields.code_verifier=verifier;res=await post(base,'/oauth/token',fields,'form');assert.equal(res.status,200);const tokens=await res.json();assert(tokens.access_token&&tokens.refresh_token);
  assert(await oauth.authenticate({headers:{authorization:'Bearer '+tokens.access_token}}));
  res=await post(base,'/oauth/token',fields,'form');assert.equal(res.status,400);
  res=await post(base,'/oauth/token',{grant_type:'refresh_token',client_id,refresh_token:tokens.refresh_token,resource:origin+'/mcp'},'form');assert.equal(res.status,200);assert((await res.json()).access_token);
  res=await post(base,'/oauth/token',{grant_type:'refresh_token',client_id,refresh_token:tokens.refresh_token,resource:origin+'/mcp'},'form');assert.equal(res.status,400);
 }finally{server.close();}
});