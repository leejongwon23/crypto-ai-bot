/**
 * Opt-in transport optimization only. This never changes the verbatim V1.7
 * policy, official snapshot, ledger, trading rules, or publication validator.
 *
 * The caller may use cachedPolicySha256 ONLY after the whole source prompt
 * has already been read in this same conversation. Unknown/mismatched hashes
 * gracefully fall back to the full unchanged response.
 */
export function applyCachedPolicyResponse(prepared, cachedPolicySha256) {
  if (!cachedPolicySha256 || prepared?.status !== 'NEEDS_AI_ANALYSIS') return prepared;
  const policy = prepared.policy;
  if (!policy || typeof policy.text !== 'string' || policy.sha256 !== cachedPolicySha256) {
    return { ...prepared, policyCache: { used: false, reason: 'HASH_NOT_MATCHED_FULL_POLICY_RETURNED' } };
  }
  const { text: _fullText, ...verbatimPolicyMetadata } = policy;
  return {
    ...prepared,
    policy: verbatimPolicyMetadata,
    policyCache: {
      used: true,
      policySha256: policy.sha256,
      prerequisite: 'Full original V1.7 policy must already have been read in this same conversation. If not, fetch get_original_v17_full_prompt before analyzing.',
      originalRulesUnchanged: true,
    },
  };
}
