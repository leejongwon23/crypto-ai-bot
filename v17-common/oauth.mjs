import crypto from 'node:crypto';
import {URL, URLSearchParams} from 'node:url';

/** OAuth2.1/PKCE for a single-owner private ChatGPT MCP plugin.
 * Login requires the server owner's existing secret; no credential is stored in URLs or source.
 */
export function makeOAuth({origin,pool,ownerSecret}) {
 if(!/^https:\/\/[^/]+$/.test(origin)) throw Error('PUBLIC_ORIGIN must be HTTPS origin');
 if(!ownerSecret || ownerSecret.length<16) throw Error('SIGNAL_API_KEY must contain at least 16 characters');
 const issuer=origin;
 const resource=origin+'/mcp';
 const scope='signals:read signals:write';
 const digest=s=>crypto.createHash('sha256').update(s).digest('hex');
 const secret=()=>crypto.randomBytes(32).toString('base64url');
 const htmlEscape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const eq=(a,b)=>{const x=Buffer.from(String(a)),y=Buffer.from(String(b));return x.length===y.length&&crypto.timingSafeEqual(x,y)};
 const sendJSON=(res,code,body,headers={})=>{res.writeHead(code,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store',...headers});res.end(JSON.stringify(body));};
 const sendHTML=(res,code,body)=>{res.writeHead(code,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store','Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",'X-Content-Type-Options':'nosniff'});res.end(body);};
 const err=(res,status,code,desc)=>sendJSON(res,status,{error:code,error_description:desc});
 const url=req=>new URL(req.url,origin);
 async function body(req){let n=0,buf=[];for await(const b of req){n+=b.length;if(n>8192)throw Error('request too large');buf.push(b);}const raw=Buffer.concat(buf).toString('utf8');if(String(req.headers['content-type']).includes('application/json'))return JSON.parse(raw);return Object.fromEntries(new URLSearchParams(raw));}
 const okScope=s=>{const a=String(s||scope).split(/\s+/).filter(Boolean);return a.length>0&&a.every(x=>['signals:read','signals:write'].includes(x))};
 const redirectOK=s=>{try{const u=new URL(s);return u.protocol==='https:'&&['chatgpt.com','chat.openai.com'].includes(u.hostname)&&!u.username&&!u.password&&!u.hash;}catch{return false}};
 const clientQuery=id=>pool.query('SELECT client_id,redirect_uris FROM v17_oauth_clients WHERE client_id=$1',[id]);
 async function init(){await pool.query(`CREATE TABLE IF NOT EXISTS v17_oauth_clients (client_id TEXT PRIMARY KEY, redirect_uris JSONB NOT NULL,created_at TIMESTAMPTZ DEFAULT NOW());
 CREATE TABLE IF NOT EXISTS v17_oauth_codes (code_hash TEXT PRIMARY KEY,client_id TEXT NOT NULL,redirect_uri TEXT NOT NULL,code_challenge TEXT NOT NULL,scope TEXT NOT NULL,expires_at TIMESTAMPTZ NOT NULL,used BOOLEAN DEFAULT FALSE);
 CREATE TABLE IF NOT EXISTS v17_oauth_access (token_hash TEXT PRIMARY KEY,client_id TEXT NOT NULL,scope TEXT NOT NULL,expires_at TIMESTAMPTZ NOT NULL);
 CREATE TABLE IF NOT EXISTS v17_oauth_refresh (refresh_hash TEXT PRIMARY KEY,client_id TEXT NOT NULL,scope TEXT NOT NULL,expires_at TIMESTAMPTZ NOT NULL,used BOOLEAN DEFAULT FALSE);`);}
 const authorizationMetadata={issuer,authorization_endpoint:origin+'/oauth/authorize',token_endpoint:origin+'/oauth/token',registration_endpoint:origin+'/oauth/register',response_types_supported:['code'],grant_types_supported:['authorization_code','refresh_token'],code_challenge_methods_supported:['S256'],token_endpoint_auth_methods_supported:['none'],scopes_supported:['signals:read','signals:write']};
 const resourceMetadata={resource,authorization_servers:[origin],scopes_supported:['signals:read','signals:write'],bearer_methods_supported:['header']};
 const challenge=`Bearer resource_metadata="${origin}/.well-known/oauth-protected-resource", scope="signals:read signals:write"`;
 async function authenticate(req){const m=/^Bearer (.+)$/.exec(String(req.headers.authorization||''));if(!m)return null;const r=await pool.query('SELECT client_id,scope FROM v17_oauth_access WHERE token_hash=$1 AND expires_at>NOW()',[digest(m[1])]);return r.rows[0]||null;}
 function loginPage(res,query,failed=false){let hidden='';for(const [k,v] of Object.entries(query))hidden+=`<input type="hidden" name="${htmlEscape(k)}" value="${htmlEscape(v)}">`;
  sendHTML(res,failed?401:200,`<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>BTC·ETH 인증</title><style>body{font-family:system-ui;background:#0b1220;color:#eee;max-width:420px;margin:80px auto;padding:18px}input,button{box-sizing:border-box;width:100%;padding:13px;margin:10px 0;border-radius:8px;border:1px solid #888;font-size:16px}button{background:#2563eb;color:white;border:0}small{color:#aaa}</style></head><body><h2>BTC·ETH 공통 분석 엔진</h2><p>ChatGPT 연결을 승인하려면 Render의 <strong>SIGNAL_API_KEY</strong> 값을 입력하세요.</p><small>이 페이지는 본인 전용 서버에서 제공하며, 비밀키를 ChatGPT 채팅에 입력하지 마세요.</small>${failed?'<p role="alert">인증키가 일치하지 않습니다.</p>':''}<form method="post" action="/oauth/authorize">${hidden}<input type="password" name="password" required autocomplete="off" placeholder="서버 인증키"><button type="submit">ChatGPT 연결 승인</button></form></body></html>`);}
 async function validateAuthRequest(p){const clientId=String(p.client_id||''),redirect=String(p.redirect_uri||''),challenge=String(p.code_challenge||'');
  if((p.resource&&p.resource!==resource)||p.response_type!=='code'||p.code_challenge_method!=='S256'||!/^[a-zA-Z0-9_-]{43,128}$/.test(challenge)||!okScope(p.scope))return {error:'invalid_request'};
  const r=await clientQuery(clientId);if(r.rows.length!==1)return {error:'invalid_client'};if(!Array.isArray(r.rows[0].redirect_uris)||!r.rows[0].redirect_uris.includes(redirect)||!redirectOK(redirect))return {error:'invalid_redirect_uri'};
  return {clientId,redirect,challenge,scope:String(p.scope||scope)};
 }
 async function issueToken(clientId,sc){const access=secret(),refresh=secret();await pool.query('INSERT INTO v17_oauth_access(token_hash,client_id,scope,expires_at) VALUES ($1,$2,$3,NOW()+INTERVAL \'1 hour\')',[digest(access),clientId,sc]);await pool.query('INSERT INTO v17_oauth_refresh(refresh_hash,client_id,scope,expires_at) VALUES($1,$2,$3,NOW()+INTERVAL \'30 days\')',[digest(refresh),clientId,sc]);return {access_token:access,token_type:'Bearer',expires_in:3600,refresh_token:refresh,scope:sc};}
 async function handle(req,res){const u=url(req),path=u.pathname;
  if(req.method==='GET'&&[ '/.well-known/oauth-authorization-server','/.well-known/oauth-authorization-server/mcp'].includes(path))return sendJSON(res,200,authorizationMetadata);
  if(req.method==='GET'&&['/.well-known/oauth-protected-resource','/.well-known/oauth-protected-resource/mcp'].includes(path))return sendJSON(res,200,resourceMetadata);
  if(req.method==='POST'&&path==='/oauth/register'){
   const p=await body(req);const uris=p.redirect_uris;if(!Array.isArray(uris)||uris.length<1||uris.length>5||!uris.every(redirectOK)||p.token_endpoint_auth_method&&p.token_endpoint_auth_method!=='none')return err(res,400,'invalid_client_metadata','Require ChatGPT HTTPS redirect and public client');
   const id='v17_'+secret();await pool.query('INSERT INTO v17_oauth_clients(client_id,redirect_uris) VALUES($1,$2)',[id,JSON.stringify(uris)]);return sendJSON(res,201,{client_id:id,client_id_issued_at:Math.floor(Date.now()/1000),redirect_uris:uris,token_endpoint_auth_method:'none',grant_types:['authorization_code','refresh_token'],response_types:['code']});
  }
  if(path==='/oauth/authorize'&&['GET','POST'].includes(req.method)){
   const p=req.method==='GET'?Object.fromEntries(u.searchParams):await body(req);const v=await validateAuthRequest(p);if(v.error)return err(res,400,v.error,'Invalid OAuth authorization request');
   if(req.method==='GET')return loginPage(res,p);
   if(!eq(p.password||'',ownerSecret))return loginPage(res,Object.fromEntries(Object.entries(p).filter(([k])=>k!=='password')),true);
   const code=secret();await pool.query('INSERT INTO v17_oauth_codes(code_hash,client_id,redirect_uri,code_challenge,scope,expires_at) VALUES($1,$2,$3,$4,$5,NOW()+INTERVAL \'5 minutes\')',[digest(code),v.clientId,v.redirect,v.challenge,v.scope]);
   const back=new URL(v.redirect);back.searchParams.set('code',code);if(typeof p.state==='string')back.searchParams.set('state',p.state);
   res.writeHead(302,{'Location':back.href,'Cache-Control':'no-store'});return res.end();
  }
  if(path==='/oauth/token'&&req.method==='POST'){
   const p=await body(req);if(p.grant_type==='authorization_code'){
    if((p.resource&&p.resource!==resource)||typeof p.code!=='string'||typeof p.code_verifier!=='string'||typeof p.client_id!=='string')return err(res,400,'invalid_request','Missing OAuth fields');
    const verifier=p.code_verifier;if(!/^[a-zA-Z0-9_~.\-]{43,128}$/.test(verifier))return err(res,400,'invalid_grant','Invalid PKCE verifier');
    const ch=crypto.createHash('sha256').update(verifier).digest('base64url');const client=await pool.connect();try{await client.query('BEGIN');const r=await client.query('SELECT client_id,redirect_uri,code_challenge,scope,expires_at,used FROM v17_oauth_codes WHERE code_hash=$1 FOR UPDATE',[digest(p.code)]);const c=r.rows[0];if(!c||c.used||new Date(c.expires_at).getTime()<Date.now()||c.client_id!==p.client_id||c.redirect_uri!==p.redirect_uri||!eq(c.code_challenge,ch)){await client.query('ROLLBACK');return err(res,400,'invalid_grant','Invalid or expired authorization code');}await client.query('UPDATE v17_oauth_codes SET used=TRUE WHERE code_hash=$1',[digest(p.code)]);await client.query('COMMIT');return sendJSON(res,200,await issueToken(c.client_id,c.scope));}catch(e){await client.query('ROLLBACK');throw e;}finally{client.release();}
   }
   if(p.grant_type==='refresh_token'){if(p.resource&&p.resource!==resource)return err(res,400,'invalid_target','Invalid resource audience');
    const client=await pool.connect();try{await client.query('BEGIN');const r=await client.query('SELECT client_id,scope,used,expires_at FROM v17_oauth_refresh WHERE refresh_hash=$1 FOR UPDATE',[digest(String(p.refresh_token||''))]);const t=r.rows[0];if(!t||t.used||new Date(t.expires_at).getTime()<Date.now()||t.client_id!==p.client_id){await client.query('ROLLBACK');return err(res,400,'invalid_grant','Invalid refresh token');}await client.query('UPDATE v17_oauth_refresh SET used=TRUE WHERE refresh_hash=$1',[digest(p.refresh_token)]);await client.query('COMMIT');return sendJSON(res,200,await issueToken(t.client_id,t.scope));}catch(e){await client.query('ROLLBACK');throw e;}finally{client.release();}
   }
   return err(res,400,'unsupported_grant_type','Use authorization_code or refresh_token');
  }
  return false;
 }
 return {init,handle,authenticate,challenge};
}