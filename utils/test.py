import torch


def get_hits(rank):
    hits = [1, 3, 10]
    total_rank = rank

    hit_results = []
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float()).item()
        hit_results.append(avg_count)
    return hit_results


def get_total_rank(test_triples, score, all_ans, eval_bz=1000, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if rel_predict==1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(_sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = _filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = _filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(_sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank

def _sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def _filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        # print(h, r, t)
        # print(ans)
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def _filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def stat_ranks(rank_list):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float()).item()
    hit_results = []
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float()).item()
        hit_results.append(avg_count)
    return mrr, hit_results
