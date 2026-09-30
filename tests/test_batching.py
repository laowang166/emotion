import unittest

from classroom_emotion.inference.batching import plan_static_batches


class StaticBatchPlannerTests(unittest.TestCase):
    def test_fifty_faces_use_batch_sixty_four(self) -> None:
        plans = plan_static_batches(50, (1, 8, 16, 32, 64))

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].compiled_batch_size, 64)
        self.assertEqual(plans[0].padding_count, 14)

    def test_counts_above_largest_bucket_are_chunked(self) -> None:
        plans = plan_static_batches(70, (1, 8, 16, 32, 64))

        self.assertEqual([(plan.item_count, plan.compiled_batch_size) for plan in plans], [(64, 64), (6, 8)])

    def test_empty_input_produces_no_inference(self) -> None:
        self.assertEqual(plan_static_batches(0, (1, 8)), ())

    def test_buckets_are_normalized(self) -> None:
        plans = plan_static_batches(9, (16, 1, 8, 8))

        self.assertEqual(plans[0].compiled_batch_size, 16)


if __name__ == "__main__":
    unittest.main()