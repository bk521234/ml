import textdistance

# Edit based
result = textdistance.hamming.distance('East Regions; Revenue: $1 - 25; Org Wgtd', 'East; Revenue: $1 - $25; Org Wgtd')
print('Hamming result: {}'.format(result))

# Token based
result = textdistance.jaccard.distance('East Regions; Revenue: $1 - 25; Org Wgtd', 'East; Revenue: $1 - $25; Org Wgtd')
print('jaccard result: {}'.format(result))

# Sequence based
result = textdistance.lcsseq.distance('East Regions; Revenue: $1 - 25; Org Wgtd', 'East; Revenue: $1 - $25; Org Wgtd')
print('lcsseq result: {}'.format(result))

# Compression based
result = textdistance.arith_ncd.distance('East Regions; Revenue: $1 - 25; Org Wgtd', 'East; Revenue: $1 - $25; Org Wgtd')
print('arith_ncd result: {}'.format(result))

result = textdistance.sqrt_ncd.distance('East Regions; Revenue: $1 - 25; Org Wgtd', 'East; Revenue: $1 - $25; Org Wgtd')
print('sqrt_ncd result: {}'.format(result))

result = textdistance.bz2_ncd.distance('East Regions; Revenue: $1 - 25; Org Wgtd', 'East; Revenue: $1 - $25; Org Wgtd')
print('bz2_ncd result: {}'.format(result))

# Phonetic
result = textdistance.mra.distance('East Regions; Revenue: $1 - 25; Org Wgtd', 'East; Revenue: $1 - $25; Org Wgtd')
print('mra result: {}'.format(result))

# Simple
result = textdistance.matrix.distance('East Regions; Revenue: $1 - 25; Org Wgtd', 'East; Revenue: $1 - $25; Org Wgtd')
print('matrix result: {}'.format(result))

