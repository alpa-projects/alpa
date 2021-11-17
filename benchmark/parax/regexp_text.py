hlo_ir = "".join(list(open("last.hlo", "r").readlines()))

import re
ids = []
for item in re.findall("all-reduce\(.*channel_id=(.*), replica_groups=\{\{0,4", hlo_ir):
    ids.append(item)

ids = "." + ".".join(ids) + "."

print(ids)
