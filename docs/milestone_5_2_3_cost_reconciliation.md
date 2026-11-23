# Milestone 5.2.3 Cost Reconciliation

The frozen full-pilot forecast was 96,550 input tokens, 12,200 output tokens, and a $0.004055100
input-token cost proxy. The session dispatched one request whose preflight estimate was 325 input
tokens, 35 output tokens, and $0.00001365.

The persistent ledger records one HTTP attempt, zero attributable tokens, and a $0.00 attributable
proxy. This is incomplete accounting: the provider response passed HTTP and provider parsing before
the Pilot A adapter rejected its probability shape, but the failure path did not persist usage or
the raw response. Actual provider billing exposure is therefore unknown for one attempt. No retry
exposure occurred.
