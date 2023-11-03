Business Intelligence Lab Submission Markdown
================
naive
4/10/2023

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [Step2 Load the dataset](#step2-load-the-dataset)
- [Step3: check for any missing data and address
  it](#step3-check-for-any-missing-data-and-address-it)
- [STEP 4. Perform EDA and Feature Selection
  —-](#step-4-perform-eda-and-feature-selection--)
  - [Compute the correlations between variables
    —-](#compute-the-correlations-between-variables--)

# Student Details

|                                                   |                                                                                                                                                                                                                                                                                                                                                             |     |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| **Student ID Numbers and Names of Group Members** | *\<list one student name, class group (just the letter; A, B, or C), and ID per line, e.g., 123456 - A - John Leposo; you should be between 2 and 5 members per group\>* \| \| 1. 135575 - B - Dennis Nzioki. \| \| 2. 134645 - B - Vivean Lydiah \| \| 3. 134765 - B - Nicholas Munene \| 4. 131653- B - Terry Joan \| \| 5. 124428 - B - Eston Gichuhi \| |     |
| **GitHub Classroom Group Name**                   | *\<specify the name of the team you created on GitHub classroom\>*                                                                                                                                                                                                                                                                                          |     |
| **Course Code**                                   | BBT4206                                                                                                                                                                                                                                                                                                                                                     |     |
| **Course Name**                                   | Business Intelligence II                                                                                                                                                                                                                                                                                                                                    |     |
| **Program**                                       | Bachelor of Business Information Technology                                                                                                                                                                                                                                                                                                                 |     |
| **Semester Duration**                             | 21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023                                                                                                                                                                                                                                                                                                |     |

# Setup Chunk

We start by installing all the required packages We start by installing
all the required packages

``` r
if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: languageserver

``` r
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: readr

``` r
## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: naniar

``` r
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: ggplot2

``` r
## corrplot ----
if (require("corrplot")) {
  require("corrplot")
} else {
  install.packages("corrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: corrplot

    ## corrplot 0.92 loaded

``` r
## ggcorrplot ----
if (require("ggcorrplot")) {
  require("ggcorrplot")
} else {
  install.packages("ggcorrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: ggcorrplot

``` r
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: caret

    ## Loading required package: lattice

``` r
## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: dplyr

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
if (!is.element("mlbench", installed.packages()[, 1])) {}
  require("mlbench") #nolint
```

    ## Loading required package: mlbench

# Step2 Load the dataset

``` r
# load the dataset PimaIndians Diabetes2 from mlbench package
data("PimaIndiansDiabetes2")

PimaIndiansDiabetes2$diabetes <- factor(PimaIndiansDiabetes2$diabetes)

str(PimaIndiansDiabetes2)
```

    ## 'data.frame':    768 obs. of  9 variables:
    ##  $ pregnant: num  6 1 8 1 0 5 3 10 2 8 ...
    ##  $ glucose : num  148 85 183 89 137 116 78 115 197 125 ...
    ##  $ pressure: num  72 66 64 66 40 74 50 NA 70 96 ...
    ##  $ triceps : num  35 29 NA 23 35 NA 32 NA 45 NA ...
    ##  $ insulin : num  NA NA NA 94 168 NA 88 NA 543 NA ...
    ##  $ mass    : num  33.6 26.6 23.3 28.1 43.1 25.6 31 35.3 30.5 NA ...
    ##  $ pedigree: num  0.627 0.351 0.672 0.167 2.288 ...
    ##  $ age     : num  50 31 32 21 33 30 26 29 53 54 ...
    ##  $ diabetes: Factor w/ 2 levels "neg","pos": 2 1 2 1 2 1 2 1 2 2 ...

``` r
dim(PimaIndiansDiabetes2)
```

    ## [1] 768   9

``` r
head(PimaIndiansDiabetes2)
```

    ##   pregnant glucose pressure triceps insulin mass pedigree age diabetes
    ## 1        6     148       72      35      NA 33.6    0.627  50      pos
    ## 2        1      85       66      29      NA 26.6    0.351  31      neg
    ## 3        8     183       64      NA      NA 23.3    0.672  32      pos
    ## 4        1      89       66      23      94 28.1    0.167  21      neg
    ## 5        0     137       40      35     168 43.1    2.288  33      pos
    ## 6        5     116       74      NA      NA 25.6    0.201  30      neg

``` r
summary(PimaIndiansDiabetes2)
```

    ##     pregnant         glucose         pressure         triceps     
    ##  Min.   : 0.000   Min.   : 44.0   Min.   : 24.00   Min.   : 7.00  
    ##  1st Qu.: 1.000   1st Qu.: 99.0   1st Qu.: 64.00   1st Qu.:22.00  
    ##  Median : 3.000   Median :117.0   Median : 72.00   Median :29.00  
    ##  Mean   : 3.845   Mean   :121.7   Mean   : 72.41   Mean   :29.15  
    ##  3rd Qu.: 6.000   3rd Qu.:141.0   3rd Qu.: 80.00   3rd Qu.:36.00  
    ##  Max.   :17.000   Max.   :199.0   Max.   :122.00   Max.   :99.00  
    ##                   NA's   :5       NA's   :35       NA's   :227    
    ##     insulin            mass          pedigree           age        diabetes 
    ##  Min.   : 14.00   Min.   :18.20   Min.   :0.0780   Min.   :21.00   neg:500  
    ##  1st Qu.: 76.25   1st Qu.:27.50   1st Qu.:0.2437   1st Qu.:24.00   pos:268  
    ##  Median :125.00   Median :32.30   Median :0.3725   Median :29.00            
    ##  Mean   :155.55   Mean   :32.46   Mean   :0.4719   Mean   :33.24            
    ##  3rd Qu.:190.00   3rd Qu.:36.60   3rd Qu.:0.6262   3rd Qu.:41.00            
    ##  Max.   :846.00   Max.   :67.10   Max.   :2.4200   Max.   :81.00            
    ##  NA's   :374      NA's   :11

# Step3: check for any missing data and address it

``` r
# Are there missing values in the dataset?
any_na(PimaIndiansDiabetes2)
```

    ## [1] TRUE

``` r
# How many?
n_miss(PimaIndiansDiabetes2)
```

    ## [1] 652

``` r
# What is the proportion of missing data in the entire dataset?
prop_miss(PimaIndiansDiabetes2)
```

    ## [1] 0.0943287

``` r
# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(PimaIndiansDiabetes2)
```

    ## # A tibble: 9 × 3
    ##   variable n_miss pct_miss
    ##   <chr>     <int>    <dbl>
    ## 1 insulin     374   48.7  
    ## 2 triceps     227   29.6  
    ## 3 pressure     35    4.56 
    ## 4 mass         11    1.43 
    ## 5 glucose       5    0.651
    ## 6 pregnant      0    0    
    ## 7 pedigree      0    0    
    ## 8 age           0    0    
    ## 9 diabetes      0    0

``` r
# Which variables contain the most missing values?
gg_miss_var(PimaIndiansDiabetes2)
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
# Which combinations of variables are missing together?
gg_miss_upset(PimaIndiansDiabetes2)
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

``` r
# Where are missing values located (the shaded regions in the plot)?
vis_miss(PimaIndiansDiabetes2) +
  theme(axis.text.x = element_text(angle = 80))
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-2-3.png)<!-- -->

``` r
PimaIndiansDiabetes2_removed_obs <-
  PimaIndiansDiabetes2 %>%
  dplyr::filter(complete.cases(.))
## OPTION 1: Remove the observations with missing values ----
# We can decide to remove all the observations that have missing values
# as follows:
PimaIndiansDiabetes2_removed_obs <- PimaIndiansDiabetes2 %>% filter(complete.cases(.))

# The initial dataset had 768 observations and 9 variables
dim(PimaIndiansDiabetes2)
```

    ## [1] 768   9

``` r
# The filtered dataset has 392 observations and 9 variables
dim(PimaIndiansDiabetes2)
```

    ## [1] 768   9

``` r
# Are there missing values in the dataset?
any_na(PimaIndiansDiabetes2_removed_obs)
```

    ## [1] FALSE

# STEP 4. Perform EDA and Feature Selection —-

## Compute the correlations between variables —-

We identify the correlated variables because it is these correlated
variables that can then be used to identify the clusters.

``` r
# Create a correlation matrix
# Option 1: Basic Table
cor(PimaIndiansDiabetes2_removed_obs[, c(1,2,3,4,5,6,7,8)]) %>%
  View()

# Option 2: Basic Plot
cor(PimaIndiansDiabetes2_removed_obs[, c(1,2,3,4,5,6,7,8)]) %>%
  corrplot(method = "square")
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
# Option 3: Fancy Plot using ggplot2
corr_matrix <- cor(PimaIndiansDiabetes2_removed_obs[, c(1,2,3,4,5,6,7,8)])

p <- ggplot2::ggplot(data = reshape2::melt(corr_matrix),
                     ggplot2::aes(Var1, Var2, fill = value)) +
  ggplot2::geom_tile() +
  ggplot2::geom_text(ggplot2::aes(label = label_wrap(label, width = 10)),
                     size = 4) +
  ggplot2::theme_minimal() +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

ggcorrplot(corr_matrix, hc.order = TRUE, type = "lower", lab = TRUE)
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-3-2.png)<!-- -->

``` r
# The correlation plot shows a -0.02 correlation between the pedigree  and the
# pressure This is worth investigating further if the intention
# of the model is to diagnose diabetes based on pedigree.

# Room_type, neighbourhood, date and other non-numeric variables and
# categorical variables are not included in the correlation, but they can be
# used as an additional dimension when plotting the scatter plot during EDA.

## Plot the scatter plots ----
# A scatter plot to show pedigree levels against pressure
# per diabetes level
ggplot(PimaIndiansDiabetes2_removed_obs,
       aes(pedigree, pressure,
           color = diabetes,
           shape = diabetes)) +
  geom_point(alpha = 0.5) +
  xlab("pedigree") +
  ylab("pressure")
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-3-3.png)<!-- -->

``` r
# A scatter plot to show pedigree levels against pressure
# per age
ggplot(PimaIndiansDiabetes2_removed_obs,
       aes(pedigree, pressure,
           color = age)) +
  geom_point(alpha = 0.5) +
  xlab("pedigree") +
  ylab("pressure")
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-3-4.png)<!-- -->

``` r
# A scatter plot to show insulin levels against age
# per diabetes level
ggplot(PimaIndiansDiabetes2_removed_obs,
       aes(insulin, age,
           color = diabetes,
           shape = diabetes)) +
  geom_point(alpha = 0.5) +
  xlab("Insulin level") +
  ylab("age")
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-3-5.png)<!-- -->

``` r
# A scatter plot to show glucose level against age 
#per diabetes levels
ggplot(PimaIndiansDiabetes2_removed_obs,
       aes(glucose, age,
           color = diabetes,
           shape = diabetes)) +
  geom_point(alpha = 0.5) +
  xlab("glucose level") +
  ylab("age")
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-3-6.png)<!-- -->

``` r
# A scatter plot to show mass against insulin levels
#  per diabetes type
ggplot(PimaIndiansDiabetes2_removed_obs,
       aes(mass, insulin,
           color = diabetes,
           shape = diabetes)) +
  geom_point(alpha = 0.5) +
  xlab("mass") +
  ylab("Insulin level")
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-3-7.png)<!-- -->

``` r
## Transform the data ----
# The K Means Clustering algorithm performs better when data transformation has
# been applied. This helps to standardize the data making it easier to compare
# multiple variables.

summary(PimaIndiansDiabetes2_removed_obs)
```

    ##     pregnant         glucose         pressure         triceps     
    ##  Min.   : 0.000   Min.   : 56.0   Min.   : 24.00   Min.   : 7.00  
    ##  1st Qu.: 1.000   1st Qu.: 99.0   1st Qu.: 62.00   1st Qu.:21.00  
    ##  Median : 2.000   Median :119.0   Median : 70.00   Median :29.00  
    ##  Mean   : 3.301   Mean   :122.6   Mean   : 70.66   Mean   :29.15  
    ##  3rd Qu.: 5.000   3rd Qu.:143.0   3rd Qu.: 78.00   3rd Qu.:37.00  
    ##  Max.   :17.000   Max.   :198.0   Max.   :110.00   Max.   :63.00  
    ##     insulin            mass          pedigree           age        diabetes 
    ##  Min.   : 14.00   Min.   :18.20   Min.   :0.0850   Min.   :21.00   neg:262  
    ##  1st Qu.: 76.75   1st Qu.:28.40   1st Qu.:0.2697   1st Qu.:23.00   pos:130  
    ##  Median :125.50   Median :33.20   Median :0.4495   Median :27.00            
    ##  Mean   :156.06   Mean   :33.09   Mean   :0.5230   Mean   :30.86            
    ##  3rd Qu.:190.00   3rd Qu.:37.10   3rd Qu.:0.6870   3rd Qu.:36.00            
    ##  Max.   :846.00   Max.   :67.10   Max.   :2.4200   Max.   :81.00

``` r
model_of_the_transform <- preProcess(PimaIndiansDiabetes2_removed_obs,
                                     method = c("scale", "center"))
print(model_of_the_transform)
```

    ## Created from 392 samples and 9 variables
    ## 
    ## Pre-processing:
    ##   - centered (8)
    ##   - ignored (1)
    ##   - scaled (8)

``` r
PimaIndiansDiabetes2_removed_obs_std <- predict(model_of_the_transform, # nolint
                                                PimaIndiansDiabetes2_removed_obs)
summary(PimaIndiansDiabetes2_removed_obs_std)
```

    ##     pregnant          glucose           pressure           triceps        
    ##  Min.   :-1.0279   Min.   :-2.1590   Min.   :-3.73423   Min.   :-2.10579  
    ##  1st Qu.:-0.7165   1st Qu.:-0.7656   1st Qu.:-0.69328   1st Qu.:-0.77454  
    ##  Median :-0.4051   Median :-0.1175   Median :-0.05308   Median :-0.01383  
    ##  Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.00000   Mean   : 0.00000  
    ##  3rd Qu.: 0.5290   3rd Qu.: 0.6601   3rd Qu.: 0.58712   3rd Qu.: 0.74689  
    ##  Max.   : 4.2657   Max.   : 2.4423   Max.   : 3.14792   Max.   : 3.21921  
    ##     insulin             mass             pedigree            age         
    ##  Min.   :-1.1953   Min.   :-2.11823   Min.   :-1.2679   Min.   :-0.9671  
    ##  1st Qu.:-0.6673   1st Qu.:-0.66683   1st Qu.:-0.7332   1st Qu.:-0.7710  
    ##  Median :-0.2571   Median : 0.01619   Median :-0.2129   Median :-0.3789  
    ##  Mean   : 0.0000   Mean   : 0.00000   Mean   : 0.0000   Mean   : 0.0000  
    ##  3rd Qu.: 0.2856   3rd Qu.: 0.57114   3rd Qu.: 0.4746   3rd Qu.: 0.5034  
    ##  Max.   : 5.8056   Max.   : 4.83999   Max.   : 5.4906   Max.   : 4.9148  
    ##  diabetes 
    ##  neg:262  
    ##  pos:130  
    ##           
    ##           
    ##           
    ## 

``` r
sapply(PimaIndiansDiabetes2_removed_obs_std[, c(1, 2, 3, 4, 5, 6, 7, 8)], sd)
```

    ## pregnant  glucose pressure  triceps  insulin     mass pedigree      age 
    ##        1        1        1        1        1        1        1        1

``` r
## Select the features to use to create the clusters ----
# OPTION 1: Use all the numeric variables to create the clusters
PimaIndiansDiabetes2_vars <-
  PimaIndiansDiabetes2_removed_obs_std[, c(1, 2, 3, 4, 5, 6, 7, 8)]
```

\#Step5 Create the clusters using the K-Means Clustering Algorithm

``` r
# We start with a random guess of the number of clusters we need
set.seed(7)
kmeans_cluster <- kmeans(PimaIndiansDiabetes2_vars, centers = 3, nstart = 20)

# We then decide the maximum number of clusters to investigate
n_clusters <- 8

# Initialize total within sum of squares error: wss
wss <- numeric(n_clusters)

set.seed(7)

# Investigate 1 to n possible clusters (where n is the maximum number of 
# clusters that we want to investigate)
for (i in 1:n_clusters) {
  # Use the K Means cluster algorithm to create each cluster
  kmeans_cluster <- kmeans(PimaIndiansDiabetes2_vars, centers = i, nstart = 20)
  # Save the within cluster sum of squares
  wss[i] <- kmeans_cluster$tot.withinss
}

## Plot a scree plot ----
# The scree plot should help you to note when additional clusters do not make
# any significant difference (the plateau).
wss_df <- tibble(clusters = 1:n_clusters, wss = wss)

scree_plot <- ggplot(wss_df, aes(x = clusters, y = wss, group = 1)) +
  geom_point(size = 4) +
  geom_line() +
  scale_x_continuous(breaks = c(2, 4, 6, 8)) +
  xlab("Number of Clusters")

scree_plot
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
# We can add guides to make it easier to identify the plateau (or "elbow").
scree_plot +
  geom_hline(
    yintercept = wss,
    linetype = "dashed",
    col = c(rep("#000000", 5), "#FF0000", rep("#000000", 2))
  )
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
# The plateau is reached at 6 clusters.
# We therefore create the final cluster with 6 clusters
# (not the initial 3 used at the beginning of this STEP.)
k <- 6
set.seed(7)
# Build model with k clusters: kmeans_cluster
kmeans_cluster <- kmeans(PimaIndiansDiabetes2_vars, centers = k, nstart = 20)

# STEP 6. Add the cluster number as a label for each observation ----
PimaIndiansDiabetes2_removed_obs$cluster_id <- factor(kmeans_cluster$cluster)

## View the results by plotting scatter plots with the labelled cluster ----
ggplot(PimaIndiansDiabetes2_removed_obs, aes(pedigree, pressure,
                                         color = cluster_id)) +
  geom_point(alpha = 0.5) +
  xlab("pedigree") +
  ylab("pressure")
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-4-3.png)<!-- -->

``` r
ggplot(PimaIndiansDiabetes2_removed_obs,
       aes(age, insulin, color = cluster_id)) +
  geom_point(alpha = 0.5) +
  xlab("age") +
  ylab("Insulin level")
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-4-4.png)<!-- -->

``` r
ggplot(PimaIndiansDiabetes2_removed_obs,
       aes(insulin, pregnant,
           color = cluster_id)) +
  geom_point(alpha = 0.5) +
  xlab("Insulin level") +
  ylab("Pregnancy level")
```

![](lab7b-clustering-submission_files/figure-gfm/unnamed-chunk-4-5.png)<!-- -->
