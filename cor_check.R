library(tidyverse)
library(rcompanion)
library(vcd)

mixed_assoc = function(df, cor_method = "pearson", adjust_cramersv_bias=TRUE) {
  
  df_comb = expand.grid(names(df), names(df),  stringsAsFactors = F) %>% set_names("X1", "X2")
  
  is_nominal = function(x) class(x) %in% c("factor", "character")
  
  is_numeric <- function(x) { is.integer(x) || is_double(x)}
  
  f = function(xName, yName) {
    x =  pull(df, xName)
    y =  pull(df, yName)
    
    result = if(is_nominal(x) && is_nominal(y)) {
        # use bias corrected cramersV as described in https://rdrr.io/cran/rcompanion/man/cramerV.html
        cv = cramerV(as.factor(x), as.factor(y), bias.correct = adjust_cramersv_bias)
        data.frame(xName, yName, assoc=cv, type="cramersV")
      
    } else if(is_numeric(x) && is_numeric(y)) {
        correlation = cor(x, y, method = cor_method)
        data.frame(xName, yName, assoc = correlation, type="correlation")
      
    } else if(is_numeric(x) && is_nominal(y)){
      # from https://stats.stackexchange.com/questions/119835/correlation-between-a-nominal-iv-and-a-continuous-dv-variable/124618#124618
        r_squared = assocstats(xtabs(~x + y))$contingency
        data.frame(xName, yName, assoc=r_squared, type="anova")
      
    } else if(is_nominal(x) && is_numeric(y)) {
        r_squared = assocstats(xtabs(~x + y))$contingency
        data.frame(xName, yName, assoc=r_squared, type="anova")
      
    } else {
        warning(paste("unmatched column type combination: ", class(x), class(y)))
    }
    # finally add complete obs number and ratio to table
    result %>% mutate(complete_obs_pairs=sum(!is.na(x) & !is.na(y)), complete_obs_ratio=complete_obs_pairs/length(x)) %>% dplyr::rename(x=xName, y=yName)
  }
  
  # apply function to each variable combination
  map2_df(df_comb$X1, df_comb$X2, f)
}
