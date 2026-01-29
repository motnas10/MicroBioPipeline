#################################################
# ---------- SAMPLES ----------
#################################################
c_ovarian = '#ffe97f'
c_gut = '#b298dc'

c_health = '#f6c3ca'
c_tumor = '#d51b30'

#################################################
# ---------- Resistome/Virulome ----------
#################################################
c_AMR = '#FFD4A3'
c_VIRULENCE = '#D4B5F0'




#################################################
# ---------- PHENOTYPES ----------
#################################################

# Traitar
c_growth        = '#ec8a83'
c_oxygen        = '#ffad85'
c_enzyme        = '#f9f176'
c_morphology    = '#8be59d'
c_proteolysis   = '#6ab4f1'
c_product       = '#a983d8'

# MOTILITY CATEGORY
c_flagellar        = '#1d2f6f'
c_chemotaxis       = '#8390fa'
c_chemotaxis_like  = '#fac748'


c_motile = "#62ac80"
c_non_motile = "#fefefe"

# ADHESION
c_adhesion   = '#156064'
c_curli      = '#00c49a'
c_fimbria    = '#f8e16c'



# -------------------------------------------------
# BIOFILM STAGE
c_initial_attachment   = '#264653'  # soft pastel pink
c_curli_fimbriae       = '#2a9d8f'
c_autoaggregation      = '#e9c46a'
c_biofilm_maturation   = '#f4a261'
c_biofilm_dispersal    = '#e76f51'
c_signaling_response   = "#969696"

# -------------------------------------------------
# UNDEFINED
c_undefined = '#f2f2f2'


#################################################
# ---------- PALETTES ----------
#################################################

source_palette = {
    'Ovarian': c_ovarian,
    'Gut': c_gut,
}

condition_palette = {
    'healthy': c_health,
    'tumor': c_tumor
}


res_vir_palette = {
    'AMR': c_AMR,
    'VIRULENCE': c_VIRULENCE,
}


traitar_cat_palette = {
    'Growth': c_growth,
    'Oxygen': c_oxygen,
    'Enzyme': c_enzyme,
    'Morphology': c_morphology,
    'Proteolysis': c_proteolysis,
    'Product': c_product,
}


motility_cat_palette = {
    'Flagellar': c_flagellar,
    'Chemotaxis': c_chemotaxis,
    'Chemotaxis-like': c_chemotaxis_like,
}

motility_palette = {
    'Motile': c_motile,
    'Non-motile': c_non_motile,

}

biofilm_stage_palette = {
    '1 - Initial attachment': c_initial_attachment,
    '2 - Curli fimbriae': c_curli_fimbriae,
    '3 - Autoaggregation': c_autoaggregation,
    '4 - Biofilm maturation': c_biofilm_maturation,
    '5 - Biofilm dispersal': c_biofilm_dispersal,
    'Signaling / Response': c_signaling_response,
}

adhesion_cat_palette = {
    'Adhesion': c_adhesion,
    'Curli': c_curli,
    'Fimbria': c_fimbria,
}

#################################################
# ---------- PALETTE LOOKUP ----------
#################################################

palette_lookup = {
    'source': source_palette,
    'condition': condition_palette,
    'res_vir': res_vir_palette,
    'motility_category': motility_cat_palette,
    'motility': motility_palette,
    'biofilm_stage': biofilm_stage_palette,
}