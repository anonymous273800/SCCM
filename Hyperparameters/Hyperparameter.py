
# WidrowHoff (LMS)
wf_learning_rate = 0.01
wf_learning_rate2 = 0.001
wf_learning_rate3 = 0.0001


# Recursive Least Squares (RLS)
rls_lambda_ = .99
rls_lambda_2 = .18
rls_delta = .01


# Online Passive-Aggressive (PA)
pa_C = .1  # reg param
pa_epsilon = .1  # epsilon is a positive parameter which controls the sensitivity
# to prediction mistakes. used to compute the loss.
pa_C2 = 0.01
pa_epsilon2 = 0.01


# OLR_WA
olr_wa_batch_size = 10
olr_wa_w_base = .5  # default value
olr_wa_w_inc = .5  # default_value

olr_wa_w_base1 = .9
olr_wa_w_inc1 = .1

olr_wa_w_base2 = .1
olr_wa_w_inc2 = .9

olr_wa_w_base_adv1 = .1
olr_wa_w_inc_adv1 = 2

olr_wa_w_base_adv2 = 4
olr_wa_w_inc_adv2 = 0.01

olr_wa_base_model_size0 = 1
olr_wa_base_model_size1 = 10
olr_wa_base_model_size2 = 2
olr_wa_increment_size = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features + 1) * 5))
olr_wa_increment_size2 = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features ) * 5))


standard_mini_batch_size1 = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features+1) * 5))
standard_mini_batch_size2 = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features) * 5))



