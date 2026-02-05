# ... inside your account loop ...

            for _ in range(n_baseline):  

                
                # 1. Generate the base 22 columns (transaction behavior)
                tx_base = [
                    acct, party_key, client_ref_date, tx_type, amt, cp,
                    round(_rand_balance(),2), _rand_party_country(),
                    0, False,
                    party_type_cd,    
                    non_nexus_flag,    
                    sample_from_transactions(txn_map, "acct_currency_cd"),   #_biased_acct_currency(),
                    _rand_transaction_key(acct, transaction_date), 
                    trans_type_cd,   
                    sample_from_transactions(txn_map, "channel_type_cd"),   #_biased_channel_type(),
                    sample_from_transactions(txn_map, "transaction_strings"),   #_biased_transaction_strings(), 
                    sample_from_transactions(txn_map, "cashier_order_flag"),   #_biased_cashier_order_flag(),
                    _rand_local_currency(amt), 
                    days_prior_review,         
                    trans_direction,     
                    transaction_date.strftime('%Y-%m-%d %H:%M:%S')
                ]
            
                # 2. Call your demographic generator for this specific party
                # This function should return a list of 20 items (Actor traits + 1.0 multipliers)
                meta_payload = _expand_demographics_for_base0_client(acct, scenario, party_key=party_key)
            
                # 3. WIDEN the row by concatenating the lists
                # Use '+' to join the two lists into one 42-member row
                full_row = tx_base + meta_payload
            
                # 4. Append the single, complete row
                rows.append(full_row)

def _expand_demographics_for_base0_client(acct, scenario=None, party_key=None, opt_overrides=None):
    
    if scenario == "structuring":
        cfg = scenario_config['structuring']
        df_cfg = cfg['demographic_factors']

        def dynamic_sample(field_name, sub_path=None):
            prob_key = f"{scenario}_{field_name}_probs"
            default_probs = cfg.get(sub_path, cfg).get(f"{field_name}{'_prob' if field_name == 'PEP' else '_probs'}")
            p_vector = opt_overrides.get(prob_key, default_probs)
            return np.random.choice(len(p_vector), p=p_vector)

        actor = {
            "PEP": np.random.binomial(1, opt_overrides.get(f"{scenario}_PEP_prob", cfg['demographics']['PEP_prob'])),
            "NatRisk":      dynamic_sample("NatRisk", "demographics"),      
            "AgeGroup":     dynamic_sample("AgeGroup", "demographics"),     
            "Occ":          dynamic_sample("Occ", "demographics"),          
            "Income":       dynamic_sample("Income", "demographics"),       
            "EntityType":   dynamic_sample("EntityType", "demographics"),   
            "Industry":     dynamic_sample("Industry", "demographics"),     
            "Channel":      dynamic_sample("channel"),                      
            "Network":      dynamic_sample("network"),                      
            "Jurisdiction": dynamic_sample("jurisdiction")                  
        }

        PEP_f      = 1.0
        NatRisk_f  = 1.0
        Occ_f      = 1.0
        Income_f   = 1.0
        Entity_f   = 1.0
        Industry_f = 1.0
        Channel_f  = 1.0
        Network_f  = 1.0
        Juris_f    = 1.0

        Total_Risk = (PEP_f * NatRisk_f * Occ_f * Income_f * Entity_f * Industry_f * Channel_f * Network_f * Juris_f)
        Total_Behavioral_Risk = max(1e-6, Total_Risk)

        metadata_payload = [
            actor["PEP"], actor["NatRisk"], actor["AgeGroup"], actor["Occ"], 
            actor["Income"], actor["EntityType"], actor["Industry"], 
            actor["Channel"], actor["Network"], actor["Jurisdiction"],
            PEP_f, NatRisk_f, Occ_f, Income_f, Entity_f, 
            Industry_f, Channel_f, Network_f, Juris_f, Total_Behavioral_Risk
        ]
        
    return metadta_payload

    elif scenario == "velocity_spike":
        cfg = scenario_config['velocity_spike']
        dw = cfg["demographics"]

        pep_prob = opt_overrides.get(f"{scenario}_PEP_prob", 0.1)
        is_pep = np.random.binomial(1, pep_prob) == 1
        
        biz_prob = opt_overrides.get(f"{scenario}_Business_prob", 0.3)
        is_biz = np.random.binomial(1, biz_prob) == 1
        
        youth_prob = opt_overrides.get(f"{scenario}_Youth_prob", 0.5)
        is_young = np.random.binomial(1, youth_prob) == 1
        
        risk_shift = opt_overrides.get(f"{scenario}_risk_skew", 1.0)
        risk_score = np.random.uniform(0, 1) ** (1 / risk_shift)

        actor = {
            "PEP": is_pep,
            "EntityType": 1 if is_biz else 0, # Mapping Business to 1
            "AgeGroup": 0 if is_young else 1,  # Mapping Young to 0
            "BaseRisk": risk_score
        }

        network_centrality_score = 1.0
        channel_hop_score        = 1.0
        savings_drain_ratio      = 1.0
       
        F_burst = 1.0
      
        F_size = 1.0

        F_inter = 1.0

        F_intra = 1.0

        F_amt = 1.0

        Total_Behavioral_Risk = F_burst * F_size * F_inter * F_intra * F_amt * network_centrality_score * channel_hop_score

        s_log_adj = opt_overrides.get(f"{scenario_type}_risk_log_base_adj", 1.0)

        compressed_total_risk = s_log_adj + np.log1p(Total_Behavioral_Risk)

        metadata_payload = [
            actor["PEP"], actor["EntityType"], actor["AgeGroup"], actor["BaseRisk"], 
             F_burst, F_size, F_inter, F_intra, F_amt, network_centrality_score, channel_hop_score, Total_Behavioral_Risk
        ]

    elif scenario == "layering":
        
        cfg = scenario_config['layering']
        dw = cfg['demographics']

        def dynamic_sample(field_name, sub_path='demographics'):
            """
            Categorical sampler that handles weight dictionaries and Optuna overrides.
            """
            prob_key = f"{scenario}_{field_name}_probs"
            
            clean_key = f"{field_name}_weights"
            
            baseline_dict = dw.get(clean_key, {})
            default_probs = list(baseline_dict.values()) if isinstance(baseline_dict, dict) else baseline_dict
        
            p_vector = opt_overrides.get(prob_key, default_probs)
            
            return np.random.choice(len(p_vector), p=p_vector)

        actor = {
            "EntityType":  dynamic_sample("entity_type"),
            "RiskScore":   dynamic_sample("risk_score"),
            "Nationality": dynamic_sample("nationality"),
            "PEP":         1 if np.random.random() < opt_overrides.get(
                               f"{scenario_type}_pep_status_prob", 
                               dw.get('pep_status_prob', 0.05)
                           ) else 0,
                           
            # --- NEW: INDUSTRY VARIATION ---
            # We sample industry index (0, 1, 2...) from optimized weights
            "Industry":    dynamic_sample("industry")
        }


        Entity_f   = 1.0
        Risk_f     = 1.0
        Juris_f    = 1.0
        PEP_f      = 1.0
        Industry_f = 1.0
        
        Static_Risk = Entity_f * Risk_f * Juris_f * PEP_f * Industry_f

        rail_switch_intensity = 1.0
        rail_div = 1.0
        centrality_boost = 1.0
        staged_escalation_factor = 1.0
        
        Total_Behavioral_Risk = (Static_Risk) * centrality_boost * (rail_switch_intensity / rail_div) * staged_escalation_factor

        metadata_payload = [
            actor["EntityType"], actor["RiskScore"], actor["Nationality"], actor["PEP"], actor["Industry"],
            Entity_f, Risk_f, Juris_f, PEP_f, Industry_f, Static_Risk,
            rail_switch_intensity, rail_div, centrality_boost, staged_escalation_factor,
            Total_Behavioral_Risk
        ]

    elif scenario == "biz_inflow_outflow_ratio":

        cfg = scenario_config['biz_inflow_outflow_ratio']
        sampling_cfg = cfg['demographic_sampling']
        multipliers = cfg['demographic_factors']  

        def dynamic_sample(field_name):

            prob_key = f"{scenario}_{field_name}_probs"
            field_cfg = sampling_cfg.get(field_name, {})
            default_probs = field_cfg.get('probabilities', [])
            p_vector = opt_overrides.get(prob_key, default_probs)
            choices = field_cfg.get('choices', [])
            return np.random.choice(choices, p=p_vector)
        
        actor = {
            "Occupation": dynamic_sample("Occupation"),
            "Industry":   dynamic_sample("Industry"),
            "EntityType": dynamic_sample("EntityType"),
            
            "Age":         opt_overrides.get(f"{scenario}_age", np.random.uniform(20, 70)),
            
            "RiskScore":   opt_overrides.get(f"{scenario}_risk_score", 
                                             np.random.randint(sampling_cfg['RiskScore']['low'], 
                                                               sampling_cfg['RiskScore']['high'] + 1)),
                                             
            "RevenueBand": opt_overrides.get(f"{scenario_type}_revenue_band", 
                                             np.random.lognormal(mean=9, sigma=1))
        }

        age          = actor["Age"]
        occupation   = actor["Occupation"]
        industry     = actor["Industry"]
        entity_type  = actor["EntityType"]
        risk_score   = actor["RiskScore"]
        revenue_band = actor["RevenueBand"]


        age_factor = 1.0
    
        occupation_risk = 1.0

        industry_factor = 1.0

        entity_multiplier = 1.0
    
        rev_factor = 1.0


        D_Risk = (occupation_risk * industry_factor * entity_multiplier * rev_factor * age_factor)

        cluster_component = 1.0

        rail_component = 1.0

        e_f = 1.0
        
        # 4. Final Calculation with 1e-6 Safeguard
        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            cluster_component *
            rail_component *
            e_f
        ))

        metadata_payload = [
            actor["Occupation"], actor["Industry"], actor["EntityType"], 
            actor["Age"], actor["RiskScore"], actor["RevenueBand"],
            age_factor, occupation_risk, industry_factor, entity_multiplier, rev_factor,
            D_Risk, cluster_component, rail_component, e_f, Total_Behavioral_Risk
        ]

    elif scenario == "biz_monthly_volume_deviation":

         def dynamic_sample(field_name):

            prob_key = f"{scenario}_{field_name}_probs"
            
            field_data = sampling_cfg.get(field_name, {})
            choices = field_data.get('choices', [])
            default_probs = field_data.get('probabilities', [])
            
            p_vector = opt_overrides.get(prob_key, default_probs)
            
            return np.random.choice(choices, p=p_vector)

        # --- 2026 RECONCILED ACTOR GENERATION ---
        actor = {

            "Age":            dynamic_sample("Age"),
            "Region":         dynamic_sample("Region"),
            "Occupation":     dynamic_sample("Occupation"),
            "BusinessSize":   dynamic_sample("BusinessSize"),
            "Sophistication": dynamic_sample("Sophistication"),

            "RiskScore": opt_overrides.get(
                f"{scenario_type}_risk_score", 
                np.random.beta(a=sampling_cfg['RiskScore']['a'], b=sampling_cfg['RiskScore']['b']) * 10
            )
        }

        # --- SAFETY UNPACK (For downstream reference) ---
        age            = actor["Age"]
        region         = actor["Region"]
        occupation     = actor["Occupation"]
        risk_score     = actor["RiskScore"]
        business_size  = actor["BusinessSize"]
        sophistication = actor["Sophistication"]

        
        age_f      = 1.0
        region_f   = 1.0
        occ_f      = 1.0
        biz_size_f = 1.0
        soph_f     = 1.0
        risk_f = 1.0
        r_div = 1.0
        
        D_Risk = (age_f * region_f * occ_f * biz_size_f * soph_f * (risk_f / r_div))

        centrality_component = 1.0
        rail_component = 1.0
        e_f = 1.0

        # Final 2026 Unified Bridge with 1e-6 Safeguard
        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            centrality_component * 
            rail_component * 
            e_f
        ))


        # Mapping the 'Who' (Gates) and the 'Logic' (Weights) for Bayesian Optimization
        metadata_payload = [
            actor["Age"], actor["Region"], actor["Occupation"], 
            actor["BusinessSize"], actor["Sophistication"], actor["RiskScore"],
            age_f, region_f, occ_f, biz_size_f, soph_f, risk_f, r_div,
            D_Risk, centrality_component, rail_component, e_f, 
            Total_Behavioral_Risk
        ]

    elif scenario == "biz_round_tripping":

        cfg = scenario_config['biz_round_tripping']
        sampling_cfg = cfg['demographic_sampling']
        multipliers = cfg['demographic_factors']

        rs_cfg = sampling_cfg.get('RiskDistribution', {'alpha': 2.0, 'beta': 5.0, 'scale': 10.0})
        raw_beta = np.random.beta(rs_cfg['alpha'], rs_cfg['beta'])
        generated_risk_score = round(raw_beta * rs_cfg['scale'], 2)

        def dynamic_sample(field_name):
            
            prob_key = f"{scenario}_{field_name}_probs"
            field_data = sampling_cfg.get(field_name, {})
            choices = field_data.get('choices', [])
            default_probs = field_data.get('probabilities', [])
            p_vector = opt_overrides.get(prob_key, default_probs)
            return np.random.choice(choices, p=p_vector)

        entity = {
            
            "EntityType":      dynamic_sample("EntityType"),
            "EntitySize":      dynamic_sample("EntitySize"),
            "OwnershipType":   dynamic_sample("OwnershipType"),
            "IndustrySector":  dynamic_sample("IndustrySector"),

            "ComplexityLevel": np.random.randint(
                opt_overrides.get(f"{scenario}_ComplexityMin", sampling_cfg['ComplexityLevel']['min']),
                opt_overrides.get(f"{scenario}_ComplexityMax", sampling_cfg['ComplexityLevel']['max']) + 1
            ),

            "RiskScore": opt_overrides.get(
                f"{scenario_type}_RiskScore_base", 
                round(np.random.beta(rs_cfg['alpha'], rs_cfg['beta']) * rs_cfg['scale'], 2)
            ),
            
            "CrossBorderFactor": opt_overrides.get(
                f"{scenario_type}_CrossBorder_f", 
                sampling_cfg.get('CrossBorderFactor', 1.0)
            )
        }


        entity_type      = entity["EntityType"]
        entity_size      = entity["EntitySize"]
        ownership_type   = entity["OwnershipType"]
        industry_sector  = entity["IndustrySector"]
        complexity_level = entity["ComplexityLevel"]
        risk_score       = entity["RiskScore"]
        cross_border_f   = entity["CrossBorderFactor"]


        entity_f   = 1.0
        size_f     = 1.0
        owner_f    = 1.0
        industry_f = 1.0
        complexity_f = 1.0
        cross_border_f = 1.0
        r_div = 1.0
        D_Risk = (entity_f * size_f * owner_f * industry_f * complexity_f * cross_border_f * (entity['RiskScore'] / r_div))

        shell_component = 1.0
        rail_component  = 1.0
        trade_component = 1.0
        econ_f = 1.0


        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            shell_component * 
            rail_component * 
            trade_component * 
            econ_f
        ))

        metadata_payload = [
            entity["EntityType"], entity["EntitySize"], entity["OwnershipType"], 
            entity["IndustrySector"], entity["ComplexityLevel"], entity["RiskScore"], 
            entity["CrossBorderFactor"],
            entity_f, size_f, owner_f, industry_f, complexity_f, cross_border_f, r_div,
            D_Risk, shell_component, rail_component, trade_component, econ_f, 
            Total_Behavioral_Risk
        ]

    elif scenario == "biz_flag_non_nexus":
    
        cfg = scenario_config['biz_flag_non_nexus']
        sampling_cfg = cfg['demographic_sampling']
        multipliers = cfg['demographic_factors']


        pfx = f"{scenario}_"

        def dynamic_sample(field_name):

            prob_key = f"{pfx}{field_name}_probs"
            
            default_probs = sampling_cfg.get(field_name, {}).get('probabilities', [])
            
            p_vector = opt_overrides.get(prob_key, default_probs)
            
            return np.random.choice(len(p_vector), p=p_vector)

        actor = {
            "Age":           dynamic_sample("Age"),
            "Occupation":    dynamic_sample("Occupation"),
            "Nationality":   dynamic_sample("Nationality"),
            "EntityType":    dynamic_sample("EntityType"),
            "RiskProfile":   dynamic_sample("RiskProfile"),
            "ActivityLevel": dynamic_sample("ActivityLevel")
        }

        
        age_f = 1.0
        occ_f = 1.0
        entity_f = 1.0
        risk_f = 1.0
        activity_f = 1.0
        nationality_f = 1.0

        D_Risk = (age_f * occ_f * entity_f * risk_f * activity_f * nationality_f)


        node_boost = 1.0
        node_base  = 1.0
        
        rail_int = 1.0
        rail_div = 1.0
        
        esc_fact = 1.0
        esc_base = 1.0

        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            (node_boost if actor['RiskProfile'] == "High" or actor['RiskProfile'] == 2 else node_base) * 
            (rail_int / rail_div) * 
            (esc_fact if actor['Nationality'] == "International" or actor['Nationality'] == 1 else esc_base)
        ))

        metadata_payload = [
            actor["Age"], actor["Occupation"], actor["Nationality"], 
            actor["EntityType"], actor["RiskProfile"], actor["ActivityLevel"],
            age_f, occ_f, entity_f, risk_f, activity_f, nationality_f,
            D_Risk, node_boost, node_base, rail_int, rail_div, 
            esc_fact, esc_base, Total_Behavioral_Risk
        ]

    elif scenario == "biz_flag_pep_indonesia":

        cfg = scenario_config['biz_flag_pep_indonesia']
        sampling_cfg = cfg['demographic_sampling']
        micro_cfg = cfg['micro_params']
        multipliers = cfg['demographic_factors']

        
        # Mandatory Dual-Prefix Handshake
        pfx = f"{scenario}_"

        def dynamic_sample(field_name):
            
            prob_key = f"{pfx}dist_{field_name}" # Matches Block 0 "dist_" prefix
            default_probs = sampling_cfg.get(field_name, {}).get('probabilities', [])
            p_vector = opt_overrides.get(prob_key, default_probs)
            
            return np.random.choice(len(p_vector), p=p_vector)


        actor = {

            "age": np.random.uniform(
                opt_overrides.get(f"{pfx}age_min", sampling_cfg['age_range'][0]),
                opt_overrides.get(f"{pfx}age_max", sampling_cfg['age_range'][1])
            ),

            "tenure": np.random.uniform(
                opt_overrides.get(f"{pfx}tenure_min", sampling_cfg['tenure_range'][0]),
                opt_overrides.get(f"{pfx}tenure_max", sampling_cfg['tenure_range'][1])
            ),

            "gender": dynamic_sample("gender"),
            "role": dynamic_sample("role"),
            "entity_type": dynamic_sample("entity_type"),

            "nationality": "Indonesia"
        }


        age = actor['age']
        gender = actor['gender']
        role = actor['role']
        tenure = actor['tenure']
        entity_type = actor['entity_type']
        
 

        role_f    = 1.0
        
        entity_f  = 1.0
        
        gender_f  = 1.0

        tenure_f = 1.0

        D_Risk = (role_f * entity_f * gender_f * tenure_f)
        

        n_boost = 1.0
        n_base  = 1.0
        
        r_int = 1.0
        r_div = 1.0

        t_decay_f   = 1.0
        t_risk_thr  = 1.0
        t_risk_base = 1.0

        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            (n_boost if actor['role'] == "Advisor" or actor['role'] == 2 else n_base) * 
            (r_int / r_div) * 
            (t_risk_base + (t_risk_base - t_decay_f) if actor['tenure'] < t_risk_thr else 1.0)
        ))

        metadata_payload = [
            # 1. Actor Persona (Categorical/Continuous DNA)
            actor["age"],           # Numeric
            actor["gender"],        # Index
            actor["role"],          # Index
            actor["tenure"],        # Numeric
            actor["entity_type"],    # Index
            actor["nationality"],   # String ("Indonesia")

            # 2. Multipliers (All Neutral 1.0 per 2026 agreement)
            role_f, 
            entity_f, 
            gender_f, 
            tenure_f,

            # 3. Intermediate Risk Calculations
            D_Risk, 

            # 4. Behavioral & Rail Params
            n_boost, 
            n_base, 
            r_int, 
            r_div, 
            t_decay_f, 
            t_risk_thr, 
            t_risk_base,

            # 5. Final Behavioral Signal
            Total_Behavioral_Risk
        ]
    elif scenario == "biz_flag_personal_to_corp":

        cfg = scenario_config['biz_flag_personal_to_corp']
        sampling_cfg = cfg['demographic_sampling']
        multipliers = cfg['demographic_factors']
        corp_cfg = cfg['corporate']
        micro_cfg = cfg['micro_params']
    

        def dynamic_sample(field_name):
            prob_key = f"{scenario}_{field_name}_probs"
            default_probs = sampling_cfg.get(field_name, {}).get('probabilities', [])
            p_vector = opt_overrides.get(prob_key, default_probs)

            return np.random.choice(len(p_vector), p=p_vector)


        actor = {

            "age": np.random.randint(
                opt_overrides.get(f"{scenario}_age_min", sampling_cfg['age_range'][0]),
                opt_overrides.get(f"{scenario}_age_max", sampling_cfg['age_range'][1]) + 1
            ),
            

            "income": np.random.lognormal(
                mean=opt_overrides.get(f"{scenario}_inc_mean", sampling_cfg['income_lognormal']['mean']),
                sigma=opt_overrides.get(f"{scenario}_inc_sigma", sampling_cfg['income_lognormal']['sigma'])
            ),
            

            "occupation":   dynamic_sample("occupation"),
            "entity_type":  dynamic_sample("entity_type"),
            "risk_profile": dynamic_sample("risk_profile")
        }


        age, occ, inc = actor['age'], actor['occupation'], actor['income']
        risk_profile, entity_type = actor['risk_profile'], actor['entity_type']


        occ_f    = 1.0
        
        entity_f = 1.0
        
        risk_f   = 1.0

        if actor['age'] < 30:
            age_f = 1.0
        elif actor['age'] > 55:
            age_f = 1.0
        else:
            age_f = 1.0

        income_f = 1.0

        D_Risk = (occ_f * entity_f * risk_f * age_f * income_f)
        
       
      
        node_boost = 1.0
        node_base  = 1.0

        rail_int = 1.0
        rail_div = 1.0
        
        esc_fact = 1.0
        esc_base = 1.0

        w_decay_f   = 1.0
        w_threshold = 1.0
        
        w_offset    = 1.0
        w_logic_sc  = 1.0
        w_inactive  = 1.0

        # --- EXECUTION: NO HARDCODED ONES, TWOS, OR HUNDREDS ---
        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            (node_boost if actor['risk_profile'] == "High" else node_base) * 
            (rail_int / rail_div) * 
            (esc_fact if actor['entity_type'] == 'BusinessLinked' else esc_base) *
            (w_offset + (w_logic_sc - w_decay_f) if actor['income'] < w_threshold else w_inactive)
        ))

        metadata_payload = [
            # 1. Actor Persona (DNA)
            actor["age"],           # Continuous (Int)
            actor["occupation"],    # Index/Categorical
            actor["income"],        # Continuous (Float)
            actor["entity_type"],   # Index/Categorical
            actor["risk_profile"],  # Index/Categorical
            
            # 2. Multipliers (Neutral 1.0)
            occ_f, 
            entity_f, 
            risk_f, 
            age_f, 
            income_f,

            # 3. Intermediate Calculations
            D_Risk, 

            # 4. Behavioral & Rail Params
            node_boost, 
            node_base, 
            rail_int, 
            rail_div, 
            esc_fact, 
            esc_base, 

            # 5. Scenario Specific Weights (Wealth/Income Logic)
            w_decay_f, 
            w_threshold, 
            w_offset, 
            w_logic_sc, 
            w_inactive,

            # 6. Final Behavioral Signal
            Total_Behavioral_Risk
        ]
