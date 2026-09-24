"""Exercise CLI and real pipeline notification dispatch together, without transport I/O."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

import main
from src.core.pipeline import StockAnalysisPipeline
from src.notification import NotificationChannel
from tests.test_pipeline_single_stock_notify import _make_result


@pytest.mark.parametrize("label,codes", [("A股", ["600519", "000001"]), ("港股", ["HK00700", "HK01810"])])
@pytest.mark.parametrize("mode", ["summary", "single", "single_cli", "merge"])
@pytest.mark.parametrize("no_notify", [False, True])
def test_cli_owns_summary_without_disabling_pipeline_modes(label, codes, mode, no_notify):
    config = SimpleNamespace(
        stock_list=codes, single_stock_notify=mode == "single",
        merge_email_notification=mode == "merge", market_review_enabled=mode == "merge",
        market_review_region="cn,hk", daily_market_context_enabled=False,
        analysis_delay=0, backtest_enabled=False, report_type="simple",
    )
    args = SimpleNamespace(
        portfolio=None, single_notify=mode == "single_cli", no_context_snapshot=True,
        no_market_review=mode != "merge", workers=2, dry_run=False,
        no_notify=no_notify, schedule=False,
    )
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = config
    pipeline.max_workers = 2
    pipeline.fetcher_manager = MagicMock()
    pipeline.analyzer = MagicMock()
    pipeline.search_service = MagicMock()
    pipeline._refresh_saved_diagnostic_snapshot = MagicMock()
    pipeline.process_single_stock = MagicMock(side_effect=lambda code, **kwargs: _make_result(code))
    notifier = pipeline.notifier = MagicMock()
    notifier.generate_aggregate_report.side_effect = lambda results, *_: "aggregate:" + ",".join(r.code for r in results)
    notifier.generate_single_stock_report.side_effect = lambda result: "single:" + result.code
    notifier.send.return_value = True
    notifier.send_to_email.return_value = True
    notifier.send_to_context.return_value = False
    notifier.get_available_channels.return_value = [NotificationChannel.EMAIL]
    notifier.get_channels_for_route.return_value = [NotificationChannel.EMAIL]
    notifier._markdown_to_image_channels = []
    notifier._should_use_image_for_channel.return_value = False
    notifier.evaluate_noise_control.return_value = SimpleNamespace(should_send=True)

    with patch.object(main, "_refresh_stock_index_cache_for_analysis"), patch.object(
        main, "_compute_trading_day_filter", return_value=(codes, "cn,hk", False)
    ), patch("src.core.pipeline.StockAnalysisPipeline", return_value=pipeline), patch.object(
        main, "_run_market_review_with_shared_lock", return_value="market review content"
    ) as market_review, patch("src.feishu_doc.FeishuDocManager") as feishu, patch.dict(
        "os.environ", {"REPORT_MARKET_LABEL": label}
    ):
        feishu.return_value.is_configured.return_value = False
        assert main.run_full_analysis(config, args, codes, raise_errors=True)

    # Real pipeline run/save/dispatch methods execute; only analysis and transports are mocked.
    assert pipeline.process_single_stock.call_count == len(codes)
    pipeline.fetcher_manager.prefetch_stock_names.assert_called_once_with(codes, use_bulk=False)
    notifier.save_report_to_file.assert_called_once()
    notifier.send_to_email.assert_not_called()  # No pipeline aggregate email in CLI modes.
    if no_notify:
        notifier.send.assert_not_called()
    elif mode in {"single", "single_cli"}:
        assert notifier.send.call_count == len(codes)
        assert {call.args[0] for call in notifier.send.call_args_list} == {"single:" + code for code in codes}
    else:
        notifier.send.assert_called_once()
        content = notifier.send.call_args.args[0]
        assert all(code in content for code in codes)
        if mode == "summary":
            assert content.startswith(f"# 📊 {label}个股分析\n")
        else:
            assert "market review content" in content
    if mode == "merge":
        assert market_review.call_args.kwargs["merge_notification"] is True
        assert market_review.call_args.kwargs["send_notification"] is (not no_notify)
    else:
        market_review.assert_not_called()
