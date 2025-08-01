from rest_framework import serializers
from .models import BrainNetwork, BrainPattern, BrainRetrieval


class BrainNetworkSerializer(serializers.ModelSerializer):
    total_patterns = serializers.SerializerMethodField()
    total_retrievals = serializers.SerializerMethodField()

    class Meta:
        model = BrainNetwork
        fields = ['id', 'name', 'embedding_dim', 'beta', 'learning_rate',
                  'merkle_root', 'created_at', 'updated_at', 'total_patterns', 'total_retrievals']

    def get_total_patterns(self, obj):
        return obj.patterns.count()

    def get_total_retrievals(self, obj):
        return obj.retrievals.count()


class BrainPatternSerializer(serializers.ModelSerializer):
    class Meta:
        model = BrainPattern
        fields = ['id', 'pattern_hash', 'text_content', 'usage_count', 'created_at'] # Added text_content


class BrainRetrievalSerializer(serializers.ModelSerializer):
    retrieved_pattern_hash = serializers.CharField(source='retrieved_pattern.pattern_hash', read_only=True)
    retrieved_text = serializers.CharField(source='retrieved_pattern.text_content', read_only=True) # Added retrieved_text

    class Meta:
        model = BrainRetrieval
        fields = ['id', 'query_text', 'retrieved_pattern_hash', 'retrieved_text', 'confidence_score',
                  'retrieval_steps_data', 'created_at'] # Changed to retrieval_steps_data


class StorePatternSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=1000)


class RetrievePatternSerializer(serializers.Serializer):
    query_text = serializers.CharField(max_length=1000)
    max_iter = serializers.IntegerField(default=10, min_value=1, max_value=50)
