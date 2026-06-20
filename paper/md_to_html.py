"""Conversor leve Markdown->HTML para gerar o PDF do paper via LibreOffice.
Uso: python paper/md_to_html.py paper/draft.md /tmp/out.html
Suporta: #/##/### , **bold**, *italic*, `code`, tabelas |..|, listas, ---.
"""
import re, sys, html
src = open(sys.argv[1], encoding='utf-8').read().split('\n')
out = ['<html><head><meta charset="utf-8"><style>',
 'body{font-family:Georgia,serif;max-width:780px;margin:40px auto;line-height:1.45;color:#111}',
 'h1{font-size:1.7em;border-bottom:2px solid #333;padding-bottom:6px}',
 'h2{font-size:1.3em;margin-top:1.4em;border-bottom:1px solid #ccc}',
 'h3{font-size:1.1em;margin-top:1.1em}',
 'table{border-collapse:collapse;margin:12px 0;width:100%}',
 'th,td{border:1px solid #999;padding:6px 10px;text-align:left;font-size:.95em}',
 'th{background:#eee}','code{background:#f2f2f2;padding:1px 4px;border-radius:3px}',
 'blockquote{border-left:3px solid #ccc;margin:0;padding-left:12px;color:#555}',
 'hr{border:none;border-top:1px solid #ccc;margin:1.5em 0}','</style></head><body>']
def inline(t):
    t=html.escape(t); t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t)
    t=re.sub(r'`(.+?)`',r'<code>\1</code>',t); t=re.sub(r'(?<!\*)\*(?!\*)(.+?)\*',r'<i>\1</i>',t)
    return t
i,n=0,len(src); para=[]
def flush():
    global para
    if para: out.append('<p>'+' '.join(inline(x) for x in para)+'</p>'); para=[]
while i<n:
    line=src[i].rstrip()
    if '|' in line and i+1<n and re.match(r'^\s*\|?[\s:|-]+\|[\s:|-]*$',src[i+1]):
        flush(); hdr=[c.strip() for c in line.strip().strip('|').split('|')]
        out.append('<table><tr>'+''.join(f'<th>{inline(c)}</th>' for c in hdr)+'</tr>'); i+=2
        while i<n and '|' in src[i]:
            cs=[c.strip() for c in src[i].strip().strip('|').split('|')]
            out.append('<tr>'+''.join(f'<td>{inline(c)}</td>' for c in cs)+'</tr>'); i+=1
        out.append('</table>'); continue
    m=re.match(r'^(#{1,3})\s+(.*)',line)
    if m: flush(); l=len(m.group(1)); out.append(f'<h{l}>{inline(m.group(2))}</h{l}>'); i+=1; continue
    if re.match(r'^>\s?(.*)',line): flush(); out.append('<blockquote>'+inline(re.match(r'^>\s?(.*)',line).group(1))+'</blockquote>'); i+=1; continue
    if re.match(r'^---+\s*$',line): flush(); out.append('<hr>'); i+=1; continue
    m=re.match(r'^\s*[-*]\s+(.*)',line)
    if m:
        flush(); it=[]
        while i<n and re.match(r'^\s*[-*]\s+(.*)',src[i]): it.append(re.match(r'^\s*[-*]\s+(.*)',src[i]).group(1)); i+=1
        out.append('<ul>'+''.join(f'<li>{inline(x)}</li>' for x in it)+'</ul>'); continue
    m=re.match(r'^\s*\d+\.\s+(.*)',line)
    if m:
        flush(); it=[]
        while i<n and re.match(r'^\s*\d+\.\s+(.*)',src[i]): it.append(re.match(r'^\s*\d+\.\s+(.*)',src[i]).group(1)); i+=1
        out.append('<ol>'+''.join(f'<li>{inline(x)}</li>' for x in it)+'</ol>'); continue
    if line.strip()=='': flush(); i+=1; continue
    para.append(line); i+=1
flush(); out.append('</body></html>')
open(sys.argv[2],'w',encoding='utf-8').write('\n'.join(out)); print("ok")
