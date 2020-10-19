import torch
import string
import asrtoolkit

class TextTransform:
    def __init__(self):
        self.char_dict = {}
        self.index_dict = {}

        self.char_dict['\''] = 0
        self.index_dict[0] = '\''
        self.char_dict[' '] = 1
        self.index_dict[1] = ' '
        for i, let in enumerate(string.ascii_lowercase):
            self.index_dict[i + 2] = let
            self.char_dict[let] = i + 2

    def text_to_int(self, text):
        labels = []
        for let in text:
            labels.append(self.char_dict[let])
        return labels

    def int_to_text(self, labels):
        text = []
        for num in labels:
            text.append(self.index_dict[num])
        return text


# argmax decoding
def decoder_func(output, answ, answ_lens, blank_label=0, del_repeated=True):
    decoded_preds, decoded_targs = [], []

    text_transform = TextTransform()

    batch_freqs = torch.argmax(output, dim=2).transpose(0, 1)

    for i, freqs in enumerate(batch_freqs):
        preds = []

        decoded_targs.append(
            text_transform.int_to_text(answ[i][:answ_lens[i]].tolist())
        )

        for j, num in enumerate(freqs):
            if num != blank_label:
                if del_repeated and j != 0 and num == freqs[j-1]:
                    continue
                preds.append(num.item())
        decoded_preds.append(text_transform.int_to_text(preds))

    return decoded_preds, decoded_targs


# beam search decoding
def beam_search_decoding(output, answ, answ_lens, blank_label=0, width=8):
    decoded_preds, decoded_targs  = [], []

    text_transform = TextTransform()

    for i, mat in enumerate(output.transpose(0, 1)):
        last = {}
        P_b, P_t = 1, 1
        P_nb = 0
        # dict       [0:prob_blank, 1:prob_not_blank, 2:prob_total]
        last[''] = [P_b, P_nb, P_t]

        for t in range(mat.shape[0]):
            curr = {}

            # sorting
            cand = [(key, el) for (key, el) in last.items()]
            sorted_cand = sorted(cand, reverse=True, key=lambda x: x[1][2]) # P_Total
            best_beams = [key for (key, el) in sorted_cand][0:width]

            for beam in best_beams:
                P_nb = 0
                if t == 0:
                    beam = ''
                else:
                    if len(beam) > 0:
                        last_num = text_transform.text_to_int(beam[-1])
                        P_nb = last[beam][1] * mat[t, last_num]

                    P_b = last[beam][2] * mat[t, blank_label]

                    if beam not in curr:
                        curr[beam] = [P_b, P_nb, P_b+P_nb]
                    else:
                        curr[beam][0] += P_b
                        curr[beam][1] += P_nb
                        curr[beam][2] += P_b + P_nb

                # 0 is blank
                for c in range(1, mat.shape[1]):
                    new_beam = beam + ''.join(text_transform.int_to_text([c]))

                    if len(beam) > 0 and last_num == c:
                        P_nb = mat[t, c] * last[beam][0]
                    else:
                        P_nb = mat[t, c] * last[beam][2]

                    if new_beam not in curr:
                        curr[new_beam] = [0, P_nb, P_nb]
                    else:
                        curr[new_beam][1] += P_nb
                        curr[new_beam][2] += P_nb
            last = curr

        cand = [(key, el) for (key, el) in last.items()]
        sorted_cand = sorted(cand, reverse=True, key=lambda x: x[1][2])
        best_beam = [x[0] for x in sorted_cand][0]

        decoded_preds.append(best_beam)

        # i - номер бача
        decoded_targs.append(
                text_transform.int_to_text(answ[i][:answ_lens[i]].tolist())
        )

    return decoded_preds, decoded_targs


def cer(target, pred):
    cer_res = asrtoolkit.cer(''.join(target), ''.join(pred))
    return cer_res


def wer(target, pred):
    wer_res = asrtoolkit.wer(''.join(target), ''.join(pred))
    return wer_res
