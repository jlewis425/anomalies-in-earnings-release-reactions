# systematic-identification-of-anomalies-in-earnings-report-reactions-in-us-stocks


### Problem Description:
The typical and expected model of stock behavior following quarterly earnings releases (henceforth referred to as "earnings events") is that companies that produce results that significantly beat expectations tend to experience positive returns and those that meaningfully miss expectations tend to experience negative returns.

However, in a significant minority of cases companies that produce earnings that meaningfully beat Wall Street consensus EPS estimates see their stocks fall substantially in value following the earnings event. These counter-intutitive reactions can be a major frustration to investors, many of whom devote substantial time and effort to research and model these quarterly results. 

### Project Description:
Apply machine learning algorithms to company financial data in order to systematically identify situations where this sort of anomalous reaction (beat estimates; experience negative returns anyway) is likely.

![alt text](https://github.com/jlewis425/anomalies-in-earnings-release-reactions/blob/master/Project_overview.png)

### Background:
Fundamental investors in equity securities devote signficant effort in evaluating and forecasting the future
prospects of the companies in which they invest. Companies which trade in the public US equity markets provide
public releases of their financials on a quarterly basis. These quarterly earnings releases are therefore 
important signposts for investors; they are the main mechanism for evaluating the operational and financial
performance of their investments. 

Companies' earnings releases typically include a quarterly income statement, other financial statements,
and additional operating metrics (in either narrative or tablular form). Most public companies in the US also host a conference call on the same day as the earnings release. During these calls corporate managers discuss the company's results and outlook and field questions from investors and Wall Street analysts that follow the company.

Quarterly earnings releases are therefore understandably much anticipated by equity investors and by the 
Wall Street analysts that follow the comapnies. Because digesting and analyzing all of the relevant details
which are typically presented in a quarterly release can be time-consuming, the marketplace has developed a 
sort of shorthand heuristic. 

Quarterly earnings per share (EPS) which exceed the Wall Street "consensus" estimate are considered "beats" and those
that fall short are considered "misses". EPS that match the consensus estimate are said to be "in line". 
In reality, the consensus EPS estimate isn't a consensus at all, but is rather the arithmetic average of all of
the analyst estimates for that company that have been published (fairly recently) for the quarter in question.

## Data Set Assembly:

### Data Sources:
The data set was assembled primarily using FactSet Research Systems' software package. FactSet's screening tool facilitates rule-based filtering of stocks and aggregation of related data. Supplementary data related to options and stock volatility was sourced from Quantcha via the Quandl internet platform. 

![alt text](https://github.com/jlewis425/anomalies-in-earnings-release-reactions/blob/master/Data_sources.png)

### Stock Sample Selection Criteria:
Stocks for the data set were selected from the universe of US equities based on the following two criteria:

* Minimum reported sales of over $100 million total over the previous four quarters; **AND**
* Minimum reported Average Daily Traded Value (stock price x trading volume) of $15 million over the previous 3 months.
















Categorization of stock market reactions to corporate earnings releases for liquid US publicly traded stocks
