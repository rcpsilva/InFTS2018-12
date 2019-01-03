import rule_base_management as rbm

rb = rbm.init_rule_base([0, 1, 2], 1)

rb[1][0] = {0, 1}
rb[1][1] = {1}
rb[1][2] = {2}

map_ = [0, 2, 4]

new_rb = rbm.update_rule_base(rb, map_, [0, 1, 2, 3, 4], 1)

print(rb)
print('=====')
print(new_rb)
