use crate::{
    error::HeisenbergError,
    search::{SearchConfig, SearchScoreAdminParams, SearchScorePlaceParams},
};

/// Builder for creating search configurations with ergonomic defaults
#[derive(Debug, Clone, Default)]
pub struct SearchConfigBuilder {
    config: SearchConfig,
}

impl SearchConfigBuilder {
    /// Create a new builder with sensible defaults
    pub fn new() -> Self {
        Self {
            config: SearchConfig::default(),
        }
    }

    /// Create a builder optimized for fast searches (fewer results, less precision)
    pub fn fast() -> Self {
        let mut builder = Self::new();
        builder.config.limit = 10;
        builder.config.max_sequential_admin_terms = 3;
        builder.config.admin_fts_search_params.limit = 20;
        builder.config.place_fts_search_params.limit = 20;
        builder
    }

    /// Create a builder optimized for comprehensive searches (more results, higher precision)
    pub fn comprehensive() -> Self {
        let mut builder = Self::new();
        builder.config.limit = 50;
        builder.config.max_sequential_admin_terms = 6;
        builder.config.place_min_importance_tier = 4;
        builder.config.admin_fts_search_params.limit = 100;
        builder.config.place_fts_search_params.limit = 100;
        builder
    }

    /// Create a builder optimized for high-quality places (major cities, landmarks)
    pub fn quality_places() -> Self {
        let mut builder = Self::new();
        builder.config.place_min_importance_tier = 2;
        builder.config.place_search_score_params.importance_weight = 0.35;
        builder.config.place_search_score_params.text_weight = 0.3;
        builder
    }

    /// Set the maximum number of results to return
    pub fn limit(mut self, limit: usize) -> Self {
        self.config.limit = limit;
        self
    }

    /// Include all data columns in results (useful for debugging)
    pub fn include_all_columns(mut self) -> Self {
        self.config.all_cols = true;
        self
    }

    /// Set the maximum number of sequential administrative terms to process
    pub fn max_admin_terms(mut self, max: usize) -> Self {
        self.config.max_sequential_admin_terms = max;
        self
    }

    /// Set the minimum importance tier for places (1=most important, 5=least important)
    pub fn place_importance_threshold(mut self, tier: u8) -> Self {
        self.config.place_min_importance_tier = tier.clamp(1, 5);
        self
    }

    /// Enable or disable proactive admin search for place candidates
    pub fn proactive_admin_search(mut self, enabled: bool) -> Self {
        self.config
            .attempt_place_candidate_as_admin_before_place_search = enabled;
        self
    }

    /// Configure text search parameters
    pub fn text_search(mut self, fuzzy: bool, limit_multiplier: usize) -> Self {
        let base_limit = self.config.limit;
        self.config.admin_fts_search_params.fuzzy_search = fuzzy;
        self.config.admin_fts_search_params.limit = base_limit * limit_multiplier;
        self.config.place_fts_search_params.fuzzy_search = fuzzy;
        self.config.place_fts_search_params.limit = base_limit * limit_multiplier;
        self
    }

    /// Configure scoring weights for administrative entities
    pub fn admin_scoring(self) -> AdminScoringBuilder {
        AdminScoringBuilder::new(self)
    }

    /// Configure scoring weights for places
    pub fn place_scoring(self) -> PlaceScoringBuilder {
        PlaceScoringBuilder::new(self)
    }

    /// Build the final configuration
    pub fn build(self) -> SearchConfig {
        self.config
    }
}

/// Builder for administrative entity scoring parameters
pub struct AdminScoringBuilder {
    parent: SearchConfigBuilder,
}

impl AdminScoringBuilder {
    fn new(parent: SearchConfigBuilder) -> Self {
        Self { parent }
    }

    /// Prioritize text matching over other factors
    pub fn prioritize_text_match(mut self) -> Self {
        self.parent.config.admin_search_score_params.text_weight = 0.6;
        self.parent.config.admin_search_score_params.pop_weight = 0.2;
        self.parent.config.admin_search_score_params.parent_weight = 0.1;
        self.parent.config.admin_search_score_params.feature_weight = 0.1;
        self
    }

    /// Prioritize population/importance over other factors
    pub fn prioritize_importance(mut self) -> Self {
        self.parent.config.admin_search_score_params.text_weight = 0.3;
        self.parent.config.admin_search_score_params.pop_weight = 0.4;
        self.parent.config.admin_search_score_params.parent_weight = 0.2;
        self.parent.config.admin_search_score_params.feature_weight = 0.1;
        self
    }

    /// Prioritize hierarchical context (parent-child relationships)
    pub fn prioritize_hierarchy(mut self) -> Self {
        self.parent.config.admin_search_score_params.text_weight = 0.3;
        self.parent.config.admin_search_score_params.pop_weight = 0.2;
        self.parent.config.admin_search_score_params.parent_weight = 0.4;
        self.parent.config.admin_search_score_params.feature_weight = 0.1;
        self
    }

    /// Set custom weights (must sum to approximately 1.0)
    pub fn custom_weights(
        mut self,
        text: f32,
        population: f32,
        parent: f32,
        feature: f32,
    ) -> Result<Self, HeisenbergError> {
        let total = text + population + parent + feature;
        if (total - 1.0).abs() > 0.1 {
            return Err(HeisenbergError::ConfigError(format!(
                "Scoring weights must sum to approximately 1.0, got {total}"
            )));
        }

        self.parent.config.admin_search_score_params = SearchScoreAdminParams {
            text_weight: text,
            pop_weight: population,
            parent_weight: parent,
            feature_weight: feature,
        };
        Ok(self)
    }

    /// Return to the main configuration builder
    pub fn done(self) -> SearchConfigBuilder {
        self.parent
    }
}

/// Builder for place scoring parameters
pub struct PlaceScoringBuilder {
    parent: SearchConfigBuilder,
}

impl PlaceScoringBuilder {
    fn new(parent: SearchConfigBuilder) -> Self {
        Self { parent }
    }

    /// Prioritize text matching over other factors
    pub fn prioritize_text_match(mut self) -> Self {
        self.parent.config.place_search_score_params.text_weight = 0.6;
        self.parent
            .config
            .place_search_score_params
            .importance_weight = 0.2;
        self.parent.config.place_search_score_params.feature_weight = 0.1;
        self.parent
            .config
            .place_search_score_params
            .parent_admin_score_weight = 0.05;
        self.parent.config.place_search_score_params.distance_weight = 0.05;
        self
    }

    /// Prioritize importance/prominence over other factors
    pub fn prioritize_importance(mut self) -> Self {
        self.parent.config.place_search_score_params.text_weight = 0.3;
        self.parent
            .config
            .place_search_score_params
            .importance_weight = 0.4;
        self.parent.config.place_search_score_params.feature_weight = 0.2;
        self.parent
            .config
            .place_search_score_params
            .parent_admin_score_weight = 0.05;
        self.parent.config.place_search_score_params.distance_weight = 0.05;
        self
    }

    /// Configure distance-based scoring with center coordinates
    pub fn with_location_bias(mut self, _lat: f32, _lon: f32, distance_weight: f32) -> Self {
        // Note: The actual center coordinates would need to be set elsewhere
        // as they're part of search parameters, not global config
        self.parent.config.place_search_score_params.distance_weight =
            distance_weight.clamp(0.0, 1.0);
        self
    }

    /// Set custom weights (must sum to approximately 1.0)
    pub fn custom_weights(
        mut self,
        text: f32,
        importance: f32,
        feature: f32,
        parent_admin: f32,
        distance: f32,
    ) -> Result<Self, HeisenbergError> {
        let total = text + importance + feature + parent_admin + distance;
        if (total - 1.0).abs() > 0.1 {
            return Err(HeisenbergError::ConfigError(format!(
                "Scoring weights must sum to approximately 1.0, got {total}"
            )));
        }

        self.parent.config.place_search_score_params = SearchScorePlaceParams {
            text_weight: text,
            importance_weight: importance,
            feature_weight: feature,
            parent_admin_score_weight: parent_admin,
            distance_weight: distance,
        };
        Ok(self)
    }

    /// Return to the main configuration builder
    pub fn done(self) -> SearchConfigBuilder {
        self.parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_builder() {
        let config = SearchConfigBuilder::new().build();
        assert_eq!(config.limit, 20);
        assert!(!config.all_cols);
    }

    #[test]
    fn test_fast_preset() {
        let config = SearchConfigBuilder::fast().build();
        assert_eq!(config.limit, 10);
        assert_eq!(config.max_sequential_admin_terms, 3);
    }

    #[test]
    fn test_comprehensive_preset() {
        let config = SearchConfigBuilder::comprehensive().build();
        assert_eq!(config.limit, 50);
        assert_eq!(config.max_sequential_admin_terms, 6);
    }

    #[test]
    fn test_method_chaining() {
        let config = SearchConfigBuilder::new()
            .limit(30)
            .include_all_columns()
            .place_importance_threshold(2)
            .admin_scoring()
            .prioritize_text_match()
            .done()
            .build();

        assert_eq!(config.limit, 30);
        assert!(config.all_cols);
        assert_eq!(config.place_min_importance_tier, 2);
        assert_eq!(config.admin_search_score_params.text_weight, 0.6);
    }

    #[test]
    fn test_custom_weights_validation() {
        let result = SearchConfigBuilder::new()
            .admin_scoring()
            .custom_weights(0.5, 0.3, 0.1, 0.05); // Sums to 0.95, should pass (within 0.1 tolerance)

        assert!(result.is_ok());

        // This should fail - too far from 1.0
        let result = SearchConfigBuilder::new()
            .admin_scoring()
            .custom_weights(0.5, 0.2, 0.1, 0.05); // Sums to 0.85, should fail

        assert!(result.is_err());
    }

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();

        assert_eq!(config.limit, 20);
        assert_eq!(config.place_min_importance_tier, 5);
        assert_eq!(config.max_sequential_admin_terms, 5);
        assert!(!config.all_cols);
    }

    #[test]
    fn test_search_config_builder_fluent_interface() {
        let config = SearchConfigBuilder::new()
            .limit(10)
            .place_importance_threshold(3)
            .max_admin_terms(7)
            .include_all_columns()
            .build();

        assert_eq!(config.limit, 10);
        assert_eq!(config.place_min_importance_tier, 3);
        assert_eq!(config.max_sequential_admin_terms, 7);
        assert!(config.all_cols);
    }

    #[test]
    fn test_search_config_builder_presets_detailed() {
        // Test fast preset values
        let fast_config = SearchConfigBuilder::fast().build();
        assert_eq!(fast_config.limit, 10);
        assert_eq!(fast_config.max_sequential_admin_terms, 3);
        assert_eq!(fast_config.admin_fts_search_params.limit, 20);
        assert_eq!(fast_config.place_fts_search_params.limit, 20);

        // Test comprehensive preset values
        let comprehensive_config = SearchConfigBuilder::comprehensive().build();
        assert_eq!(comprehensive_config.limit, 50);
        assert_eq!(comprehensive_config.max_sequential_admin_terms, 6);
        assert_eq!(comprehensive_config.place_min_importance_tier, 4);
        assert_eq!(comprehensive_config.admin_fts_search_params.limit, 100);
        assert_eq!(comprehensive_config.place_fts_search_params.limit, 100);

        // Test quality places preset values
        let quality_config = SearchConfigBuilder::quality_places().build();
        assert_eq!(quality_config.place_min_importance_tier, 2);
        assert_eq!(
            quality_config.place_search_score_params.importance_weight,
            0.35
        );
        assert_eq!(quality_config.place_search_score_params.text_weight, 0.3);
    }

    #[test]
    fn test_search_config_builder_chaining() {
        // Test that methods can be chained in different orders
        let config1 = SearchConfigBuilder::new()
            .limit(15)
            .place_importance_threshold(2)
            .build();

        let config2 = SearchConfigBuilder::new()
            .place_importance_threshold(2)
            .limit(15)
            .build();

        // Both should have the same values regardless of order
        assert_eq!(config1.limit, config2.limit);
        assert_eq!(
            config1.place_min_importance_tier,
            config2.place_min_importance_tier
        );
    }

    #[test]
    fn test_search_config_builder_override_presets() {
        // Test that preset values can be overridden
        let config = SearchConfigBuilder::fast()
            .limit(100) // Override the fast preset limit
            .place_importance_threshold(1) // Override the fast preset threshold
            .build();

        assert_eq!(config.limit, 100); // Should use overridden value
        assert_eq!(config.place_min_importance_tier, 1); // Should use overridden value
        assert_eq!(config.max_sequential_admin_terms, 3); // Should keep fast preset value
    }

    #[test]
    fn test_search_config_clone() {
        let original = SearchConfigBuilder::new()
            .limit(25)
            .place_importance_threshold(4)
            .max_admin_terms(6)
            .include_all_columns()
            .build();

        let cloned = original.clone();

        assert_eq!(original.limit, cloned.limit);
        assert_eq!(
            original.place_min_importance_tier,
            cloned.place_min_importance_tier
        );
        assert_eq!(
            original.max_sequential_admin_terms,
            cloned.max_sequential_admin_terms
        );
        assert_eq!(original.all_cols, cloned.all_cols);
    }

    #[test]
    fn test_edge_case_values() {
        // Test with minimum reasonable values
        let config = SearchConfigBuilder::new()
            .limit(1) // Minimum limit
            .place_importance_threshold(1) // Minimum threshold
            .max_admin_terms(1) // Minimum admin terms
            .build();

        assert_eq!(config.limit, 1);
        assert_eq!(config.place_min_importance_tier, 1);
        assert_eq!(config.max_sequential_admin_terms, 1);

        // Test with high values
        let config_high = SearchConfigBuilder::new()
            .limit(1000)
            .place_importance_threshold(5)
            .max_admin_terms(20)
            .build();

        assert_eq!(config_high.limit, 1000);
        assert_eq!(config_high.place_min_importance_tier, 5);
        assert_eq!(config_high.max_sequential_admin_terms, 20);
    }

    #[test]
    fn test_place_importance_configuration() {
        for threshold in 1..=5 {
            let config = SearchConfigBuilder::new()
                .place_importance_threshold(threshold)
                .build();

            assert_eq!(config.place_min_importance_tier, threshold);
        }
    }

    #[test]
    fn test_include_all_columns() {
        // Default should be false
        let default_config = SearchConfigBuilder::new().build();
        assert!(!default_config.all_cols);

        // When enabled, should be true
        let all_columns_config = SearchConfigBuilder::new().include_all_columns().build();
        assert!(all_columns_config.all_cols);

        // Should work with other settings
        let combined_config = SearchConfigBuilder::new()
            .limit(10)
            .include_all_columns()
            .place_importance_threshold(3)
            .build();
        assert!(combined_config.all_cols);
        assert_eq!(combined_config.limit, 10);
        assert_eq!(combined_config.place_min_importance_tier, 3);
    }
}
