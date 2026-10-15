"""Identity of ELF loadable content, excluding non-loaded source/debug paths."""
import hashlib,struct

def sections(data):
    if data[:6]!=b'\x7fELF\x02\x01':raise ValueError('Expected little-endian ELF64')
    offset=struct.unpack_from('<Q',data,40)[0]
    entry,count,strings=struct.unpack_from('<HHH',data,58)
    if entry!=64 or not 0<strings<count or offset+count*entry>len(data):raise ValueError('Invalid section table')
    table=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*entry) for i in range(count)]
    names=table[strings];names=data[names[4]:names[4]+names[5]]
    result={}
    for row in table:
        end=names.find(b'\0',row[0]);name=names[row[0]:end].decode()
        if row[1]!=8 and row[4]+row[5]>len(data):raise ValueError('Section out of bounds')
        result[name]=dict(type=row[1],flags=row[2],address=row[3],size=row[5],align=row[8],
            bytes=b'' if row[1]==8 else data[row[4]:row[4]+row[5]])
    return result

def loaded_identity(data):
    h=hashlib.sha256();h.update(data[:16]);h.update(data[16:24]);h.update(data[48:52])
    selected={k:v for k,v in sections(data).items() if v['flags']&2}
    if '.text' not in selected:raise ValueError('No executable text')
    for name,row in sorted(selected.items()):
        h.update(name.encode()+b'\0')
        h.update(struct.pack('<QQQQQ',row['type'],row['flags'],row['address'],row['size'],row['align']))
        h.update(row['bytes'])
    return h.hexdigest()
