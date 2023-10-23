# loan_recovery
Model predicting loan recovery based on customer behaviour

# Data Source:

https://drive.google.com/drive/folders/1N1Kr9wSZVRJkwrMUkXMLoAsuaPAh9Yly?usp=share_link

# Data description:

Customer Information:

customer_id: Unique identifier for each customer application.

firstname: First name of the customer.

lastname: Last name of the customer.

Credit Record Information:

record_number: Sequence number of the credit product in the credit history.

days_since_opened: Days from credit opening date to data collection date.

days_since_confirmed: Days from credit information confirmation date till data collection date.

primary_term: Planned number of days from credit opening date to closing date.

final_term: Actual number of days from credit opening date to closing date.

days_till_primary_close: Planned number of days from data collection date until loan closing date.

days_till_final_close: Actual number of days from data collection date until loan closing date.

loans_credit_limit: Credit limit for the customer's loans.

loans_next_payment_summary: Amount of the next loan payment.

loans_outstanding_balance: Outstanding balance amount.

loans_max_overdue_amount: Maximum overdue amount.

loans_credit_cost_rate: Cost rate associated with loans.

Loan Overdue Information:

loans_within_5_days to loans_over_90_days: Number of loans overdue within different time frames.

is_zero_loans_within_5_days to is_zero_loans_over_90_days: Binary indicators for zero loans overdue within different time frames.

Credit Utilization and Limit Information:

utilization: Credit utilization rate.
over_limit_count: Count of instances where the customer went over the credit limit.

max_over_limit_count: Maximum count of instances where the customer exceeded the credit limit.

is_zero_utilization: Binary indicator if credit utilization rate is zero.

is_zero_over_limit_count: Binary indicator if the count of over limit instances is zero.

is_zero_max_over_limit_count: Binary indicator if the maximum over limit count is zero.

Encoded Features:

encoded_payment_X: Encoded information about payment X (categorical features converted to numerical values).

encoded_loans_account_holder_type: Encoded information about the type of account holder for loans.

encoded_loans_credit_status: Encoded information about the credit status of loans.

encoded_loans_credit_type: Encoded information about the type of credit for loans.

encoded_loans_account_currency: Encoded information about the currency used for loans.

Close Flags:

primary_close_flag: Binary indicator for primary term close.
final_close_flag: Binary indicator for final term close.
