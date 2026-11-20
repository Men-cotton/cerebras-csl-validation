"""Construct or publish one discoverable watcher commit through GitHub’s structured Git Data API."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile

import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from csl_validation.watch import trailers


def verify(repo, revision):
    tip=git(repo,'rev-parse',revision+'^{commit}').decode().strip()
    parent=git(repo,'rev-parse',tip+'^').decode().strip()
    delta=git(repo,'diff','--diff-filter=A','--name-only','-z',parent,tip,'--','requests').decode().split('\0')
    new=[p for p in delta if p.endswith('.json')]
    if len(new)!=1:
        raise ValueError('a hardware request commit must introduce exactly one manifest')
    path=new[0]
    rid=json.loads(git(repo,'show',tip+':'+path))['id']
    if trailers(git(repo,'show','-s','--format=%B',tip).decode())!=(rid,path):
        raise ValueError('the final commit paragraph must register the exact request ID and manifest')
    return dict(sha=tip,parent=parent,id=rid,manifest=path,manifest_blob=git(repo,'rev-parse',tip+':'+path).decode().strip())



def git(repo, *args):
    return subprocess.check_output(['git','-C',repo,*args])


def save(path, value):
    path.write_text(json.dumps(value,separators=(',',':')),encoding='utf-8')


def identity(repo, revision, fmt):
    name,email,date=git(repo,'show','-s','--format='+fmt,revision).decode().rstrip('\n').split('\0')
    return dict(name=name,email=email,date=date)


def plan(repo, revision, out):
    reg=verify(repo,revision)
    tip,parent=reg['sha'],reg['parent']
    out.mkdir(parents=True,exist_ok=False)
    save(out/'registration.json',reg)
    changes=git(repo,'diff-tree','--no-commit-id','--raw','--no-abbrev','--no-renames','-r','-z',parent,tip).split(b'\0')
    known={v.split(b' ',1)[0].decode() for v in git(repo,'rev-list','--objects',parent).splitlines()}
    entries=[]
    for line,path in zip(changes[:-1:2],changes[1::2],strict=True):
        _,mode,_,oid,status=line.decode().split()
        if status not in ('A','M') or mode not in ('100644','100755'):
            raise ValueError('a watcher publication requires regular added or modified files')
        entry=dict(path=path.decode(),mode=mode,type='blob')
        data=git(repo,'cat-file','blob',oid)
        if oid in known:
            entry['sha']=oid
        else:
            try:
                text=data.decode('utf-8')
                if '\0' in text:
                    raise ValueError('binary')
                entry['content']=text
            except (ValueError,UnicodeDecodeError):
                entry['sha']=oid
                if not (out/(oid+'.json')).is_file():
                    save(out/(oid+'.json'),dict(encoding='base64',content=base64.b64encode(data).decode()))
        entries.append(entry)
    save(out/'tree.json',dict(base_tree=git(repo,'rev-parse',parent+'^{tree}').decode().strip(),tree=entries))
    message=git(repo,'cat-file','commit',tip).split(b'\n\n',1)[1].decode()
    commit=dict(message=message,tree=git(repo,'rev-parse',tip+'^{tree}').decode().strip(),parents=[parent],
                author=identity(repo,tip,'%an%x00%ae%x00%aI'),committer=identity(repo,tip,'%cn%x00%ce%x00%cI'))
    save(out/'commit.json',commit)
    return reg


def publish(repo, revision, upstream):
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'plan'
        reg=plan(repo,revision,out)
        commit=json.loads((out/'commit.json').read_bytes())
        def api(path,*options):
            return json.loads(subprocess.check_output(['gh','api','repos/'+upstream+'/'+path,*options]))
        def checked(method,path,body,expected):
            actual=api(path,'-X',method,'--input',str(body))['sha']
            if actual!=expected:
                raise ValueError('unexpected remote Git object: '+actual)
        def check_remote():
            record=api('git/commits/'+reg['sha'])
            if (record['tree']['sha']!=commit['tree'] or [v['sha'] for v in record['parents']]!=[reg['parent']]
                    or record['message'].rstrip('\n')!=commit['message'].rstrip('\n')):
                raise ValueError('remote commit content or registration trailers changed')
        remote=api('git/ref/heads/main')['object']['sha']
        if remote==reg['sha']:
            check_remote()
        elif remote==reg['parent']:
            for bin_file in sorted(out.glob('[0-9a-f]'*40+'.json')):
                checked('POST','git/blobs',bin_file,bin_file.stem)
            checked('POST','git/trees',out/'tree.json',commit['tree'])
            checked('POST','git/commits',out/'commit.json',reg['sha'])
            check_remote()
            ref=out/'ref.json'
            save(ref,dict(sha=reg['sha'],force=False))
            if api('git/refs/heads/main','-X','PATCH','--input',str(ref))['object']['sha']!=reg['sha']:
                raise ValueError('remote ref update did not identify the request')
        else:
            raise ValueError('watcher main advanced; reconcile against its actual head')
        if (api('git/ref/heads/main')['object']['sha']!=reg['sha'] or
                api('contents/'+reg['manifest'],'-X','GET','-f','ref='+reg['sha'])['sha']!=reg['manifest_blob']):
            raise ValueError('remote ref or frozen request blob changed')
        return reg


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','publish'])
    p.add_argument('repo',type=Path)
    p.add_argument('revision')
    p.add_argument('--out',type=Path)
    p.add_argument('--upstream',default='Men-cotton/cerebras-csl-validation')
    a=p.parse_args()
    if a.action=='plan':
        if not a.out:
            p.error('plan requires --out')
        result=plan(a.repo.resolve(),a.revision,a.out)
    else:
        result=publish(a.repo.resolve(),a.revision,a.upstream)
    print(json.dumps(result))
